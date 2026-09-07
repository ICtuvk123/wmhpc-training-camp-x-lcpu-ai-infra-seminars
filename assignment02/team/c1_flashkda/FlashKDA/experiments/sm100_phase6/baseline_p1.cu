// P1: S[128,128] + K^T[128,16] @ U[16,128] -> C[128,128].
//
// The kernel uses the same SM80 MMA atom and Phase-6 warp-to-tile mapping as K2,
// but remains independent of the production kernel.

#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>

#include "../../csrc/smxx/utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using BF16 = cutlass::bfloat16_t;

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 16;
constexpr int kThreads = 128;

using KtSmemLayout = decltype(cute::tile_to_shape(
    cute::GMMA::Layout_MN_INTER_Atom<BF16>{},
    cute::make_shape(cute::Int<kM>{}, cute::Int<kK>{}), cute::LayoutRight{}));
using USmemLayout = decltype(cute::tile_to_shape(
    cute::GMMA::Layout_K_INTER_Atom<BF16>{},
    cute::make_shape(cute::Int<kN>{}, cute::Int<kK>{}), cute::LayoutRight{}));

struct BaselineSharedStorage {
  alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<KtSmemLayout>> kt;
  alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<USmemLayout>> u;
};

constexpr int kDynamicSmemBytes = sizeof(BaselineSharedStorage);

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t status_ = (expr);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status_));                                 \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// Layout contract:
//   kt[m * kK + k]  is row-major [128, 16]
//   u [k * kN + n]  is row-major [16, 128]
//   c [batch * kM * kN + m * kN + n] is FP32 row-major output
__global__ void baseline_p1_kernel(const BF16* __restrict__ kt,
                                   const BF16* __restrict__ u,
                                   const float* __restrict__ state,
                                   float* __restrict__ c) {
  using namespace cute;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto& storage = *reinterpret_cast<BaselineSharedStorage*>(smem_raw);
  Tensor s_kt = make_tensor(make_smem_ptr(storage.kt.begin()), KtSmemLayout{});
  Tensor s_u = make_tensor(make_smem_ptr(storage.u.begin()), USmemLayout{});

  Tensor g_kt = make_tensor(
      make_gmem_ptr(kt),
      make_layout(make_shape(Int<kM>{}, Int<kK>{}), LayoutRight{}));
  // MMA B uses logical (N,K), even though the allocation is row-major U[K,N].
  // Copy this transposed view into K-major SMEM for the non-transposing LDSM.
  Tensor g_u = make_tensor(
      make_gmem_ptr(u),
      make_layout(make_shape(Int<kN>{}, Int<kK>{}),
                  make_stride(Int<1>{}, Int<kN>{})));

  cooperative_copy<kThreads>(threadIdx.x, g_kt, s_kt);
  cooperative_copy<kThreads>(threadIdx.x, g_u, s_u);
  __syncthreads();

  auto mma = make_tiled_mma(
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
      Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  ThrMMA thr_mma = mma.get_slice(lane_id);

  auto smem_tiled_copy_a =
      make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
  auto smem_thr_copy_a = smem_tiled_copy_a.get_thread_slice(lane_id);
  auto smem_tiled_copy_b =
      make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
  auto smem_thr_copy_b = smem_tiled_copy_b.get_thread_slice(lane_id);

  Tensor a_ref = local_tile(
      s_kt, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
  Tensor b_ref = local_tile(
      s_u, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

  Tensor tCrAi_a =
      make_fragment_like<BF16>(thr_mma.partition_fragment_A(a_ref));
  Tensor tCrA = thr_mma.partition_fragment_A(a_ref);
  auto tCrAi_a_view = smem_thr_copy_a.retile_D(tCrAi_a);

  Tensor tCrBi_b =
      make_fragment_like<BF16>(thr_mma.partition_fragment_B(b_ref));
  Tensor tCrB = thr_mma.partition_fragment_B(b_ref);
  auto tCrBi_b_view = smem_thr_copy_b.retile_D(tCrBi_b);

  const float* block_s = state + static_cast<size_t>(blockIdx.x) * kM * kN;
  Tensor g_s = make_tensor(make_gmem_ptr(block_s),
      make_layout(make_shape(Int<kM>{}, Int<kN>{}), LayoutRight{}));
  float* block_c = c + static_cast<size_t>(blockIdx.x) * kM * kN;
  Tensor g_c = make_tensor(
      make_gmem_ptr(block_c),
      make_layout(make_shape(Int<kM>{}, Int<kN>{}), LayoutRight{}));

#pragma unroll
  for (int m_block = 0; m_block < kM / 16; ++m_block) {
    Tensor a_block = local_tile(
        s_kt, make_shape(Int<16>{}, Int<16>{}), make_coord(m_block, 0));
    copy(smem_tiled_copy_a, smem_thr_copy_a.partition_S(a_block),
         tCrAi_a_view);
    cute::transform(tCrAi_a, tCrA, cute::identity{});

#pragma unroll
    for (int block_in_warp = 0; block_in_warp < 2; ++block_in_warp) {
      const int n_block = warp_id * 2 + block_in_warp;
      Tensor b_block = local_tile(
          s_u, make_shape(Int<16>{}, Int<16>{}), make_coord(n_block, 0));
      copy(smem_tiled_copy_b, smem_thr_copy_b.partition_S(b_block),
           tCrBi_b_view);
      cute::transform(tCrBi_b, tCrB, cute::identity{});

      Tensor c_block = local_tile(
          g_c, make_shape(Int<16>{}, Int<16>{}),
          make_coord(m_block, n_block));
      Tensor tCgC = thr_mma.partition_C(c_block);
      Tensor tCrC = thr_mma.make_fragment_C(tCgC);
      Tensor state_block = local_tile(g_s, make_shape(Int<16>{}, Int<16>{}),
                                      make_coord(m_block, n_block));
      copy(thr_mma.partition_C(state_block), tCrC);
      gemm(thr_mma, tCrA(_, _, Int<0>{}), tCrB(_, _, Int<0>{}), tCrC);
      copy(tCrC, tCgC);
    }
  }
}

#include "p1_runner.cuh"

int main(int argc, char** argv) {
  auto launch = [](int batch, const BF16* kt, const BF16* u, const float* s, float* c) {
    baseline_p1_kernel<<<batch, kThreads, kDynamicSmemBytes>>>(kt, u, s, c);
  };
  return run_p1(argc, argv, "baseline", baseline_p1_kernel, launch, 0);
}
