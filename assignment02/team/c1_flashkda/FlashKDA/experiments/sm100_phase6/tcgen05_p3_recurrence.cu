// P3: persistent FP32 register state; TMEM holds only each step's K^T U.
//
// This follows CUTLASS examples/cute/tutorial/blackwell/01_mma_sm100.cu, reduced
// to the exact Phase-6 shape and a 128-column TMEM allocation.

#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using BF16 = cutlass::bfloat16_t;

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 16;
constexpr int kThreads = 128;
constexpr int kTmemColumns = 128;

CUTE_HOST_DEVICE constexpr auto make_tcgen05_mma() {
  using namespace cute;
  return make_tiled_mma(
      SM100_MMA_F16BF16_SS<BF16, BF16, float, kM, kN,
                           UMMA::Major::K, UMMA::Major::K>{});
}

CUTE_HOST_DEVICE constexpr auto make_tcgen05_tiler() {
  return cute::make_shape(cute::Int<kM>{}, cute::Int<kN>{},
                          cute::Int<kK>{});
}

CUTE_HOST_DEVICE constexpr auto make_tcgen05_a_smem_layout() {
  using namespace cute;
  auto mma = make_tcgen05_mma();
  auto shape = partition_shape_A(
      mma, make_shape(Int<kM>{}, Int<kK>{}));
  // K=16 is one BF16 core matrix. SW32's 16-element K extent divides this
  // exact tile; SW64/SW128 require larger K tiles.
  return UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<BF16>{}, shape);
}

CUTE_HOST_DEVICE constexpr auto make_tcgen05_b_smem_layout() {
  using namespace cute;
  auto mma = make_tcgen05_mma();
  auto shape = partition_shape_B(
      mma, make_shape(Int<kN>{}, Int<kK>{}));
  return UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<BF16>{}, shape);
}

using ASmemLayout = decltype(make_tcgen05_a_smem_layout());
using BSmemLayout = decltype(make_tcgen05_b_smem_layout());

struct Tcgen05SharedStorage {
  alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<ASmemLayout>> a;
  alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BSmemLayout>> b;
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_a() {
    return cute::make_tensor(cute::make_smem_ptr(a.begin()), ASmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_b() {
    return cute::make_tensor(cute::make_smem_ptr(b.begin()), BSmemLayout{});
  }
};

constexpr int kDynamicSmemBytes = sizeof(Tcgen05SharedStorage);

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t status_ = (expr);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status_));                                 \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// kt[steps,M,K], u[steps,K,N] are shared across CTAs, as in P2.
// state[batch,M,N], g_total[steps,batch,M], c[batch,M,N] are FP32.
__global__ void tcgen05_p3_recurrence_kernel(const BF16* __restrict__ kt,
                                  const BF16* __restrict__ u,
                                  const float* __restrict__ state,
                                  const float* __restrict__ g_total,
                                  float* __restrict__ c, int steps) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
  using namespace cute;

  auto tiled_mma = make_tcgen05_mma();
  auto mma_tiler = make_tcgen05_tiler();

  float* block_c = c + static_cast<size_t>(blockIdx.x) * kM * kN;
  Tensor m_c = make_tensor(
      make_gmem_ptr(block_c),
      make_layout(make_shape(Int<kM>{}, Int<kN>{}),
                  make_stride(Int<kN>{}, Int<1>{})));

  auto mma_coord = make_coord(0, 0, _);
  Tensor g_c = local_tile(m_c, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto& storage = *reinterpret_cast<Tcgen05SharedStorage*>(smem_raw);
  Tensor s_a = storage.tensor_a();
  Tensor s_b = storage.tensor_b();

  ThrMMA cta_mma = tiled_mma.get_slice(Int<0>{});
  Tensor tCgC = cta_mma.partition_C(g_c);
  Tensor tCrA = cta_mma.make_fragment_A(s_a);
  Tensor tCrB = cta_mma.make_fragment_B(s_b);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

  const uint32_t elected_thread = cute::elect_one_sync();
  const bool elected_warp = threadIdx.x / 32 == 0;
  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (elected_warp) {
    tmem_allocator.allocate(kTmemColumns, &storage.tmem_base_ptr);
  }
  __syncthreads();
  if (elected_warp) {
    tmem_allocator.release_allocation_lock();
  }
  tCtAcc.data() = storage.tmem_base_ptr;

  if (elected_warp && elected_thread) {
    cute::initialize_barrier(storage.mma_barrier, 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  TiledCopy tmem_to_register =
      make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy tmem_copy_thread = tmem_to_register.get_slice(threadIdx.x);
  Tensor tDtAcc = tmem_copy_thread.partition_S(tCtAcc);
  Tensor tDgC = tmem_copy_thread.partition_D(tCgC);
  using AccType = typename decltype(tCtAcc)::value_type;
  // Preserve copy-atom mode 0: TMEM's source mode describes 32 data paths,
  // whereas each thread's destination mode contains one value.
  Tensor tDtAccGrouped = group_modes<1, decltype(rank(tDtAcc))::value>(tDtAcc);

  // Use the output partition so each S value matches its accumulator value.
  // S never enters TMEM; the MMA retains P0's Zero -> One accumulation.
  const float* block_s = state + static_cast<size_t>(blockIdx.x) * kM * kN;
  Tensor m_s = make_tensor(make_gmem_ptr(block_s), m_c.layout());
  Tensor g_s = local_tile(m_s, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor tCgS = cta_mma.partition_C(g_s);
  Tensor tDgS = tmem_copy_thread.partition_D(tCgS);
  Tensor tDrS = make_tensor<AccType>(shape(tDgS));
  Tensor tDrSGrouped = group_modes<1, decltype(rank(tDrS))::value>(tDrS);
  CUTE_STATIC_ASSERT_V(size<0>(tDrSGrouped) == Int<1>{});
  // The only initial-state load. This fragment lives across every step.
  copy(tDgS, tDrS);

  // A runtime loop keeps the same compiled register footprint for all N.
#pragma unroll 1
  for (int step = 0; step < steps; ++step) {
    Tensor m_a = make_tensor(
        make_gmem_ptr(kt + static_cast<size_t>(step) * kM * kK),
        make_layout(make_shape(Int<kM>{}, Int<kK>{}),
                    make_stride(Int<kK>{}, Int<1>{})));
    // B is the transposed [N,K] view of row-major U[K,N], just as in P2.
    Tensor m_b = make_tensor(
        make_gmem_ptr(u + static_cast<size_t>(step) * kK * kN),
        make_layout(make_shape(Int<kN>{}, Int<kK>{}),
                    make_stride(Int<1>{}, Int<kN>{})));
    Tensor g_a = local_tile(m_a, mma_tiler, mma_coord, Step<_1, X, _1>{});
    Tensor g_b = local_tile(m_b, mma_tiler, mma_coord, Step<X, _1, _1>{});
    Tensor tCgA = cta_mma.partition_A(g_a);
    Tensor tCgB = cta_mma.partition_B(g_b);
    cooperative_copy<kThreads>(threadIdx.x, tCgA(_, _, _, Int<0>{}), s_a);
    cooperative_copy<kThreads>(threadIdx.x, tCgB(_, _, _, Int<0>{}), s_b);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    // Reset every step: A_t is a fresh product, never a TMEM state update.
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    if (elected_warp) {
#pragma unroll
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&storage.mma_barrier);
    }
    // One commit per step; the initialized barrier starts at phase zero.
    cute::wait_barrier(storage.mma_barrier, step & 1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    // Production fwd_kernel2.cuh Phase 6 loads g_total(m*16 + group_id)
    // and g_total(m*16 + group_id + 8) for the two accumulator rows.
    // Its s_acc_T / K_restored^T U view has key-feature rows and value-feature
    // columns. In this row-major [M,N] experiment: C[m,n] = S[m,n]*g[m] + KU[m,n].
    // Broadcast stride (1,0), then reuse the SAME output partition: no assumed
    // mapping between a Blackwell TMEM-copy lane and a production SM80 lane.
    const float* block_g = g_total +
        (static_cast<size_t>(step) * gridDim.x + blockIdx.x) * kM;
    Tensor m_g = make_tensor(make_gmem_ptr(block_g),
        make_layout(make_shape(Int<kM>{}, Int<kN>{}),
                    make_stride(Int<1>{}, Int<0>{})));
    Tensor g_g = local_tile(m_g, mma_tiler, mma_coord, Step<_1, _1, X>{});
    Tensor tCgG = cta_mma.partition_C(g_g);
    Tensor tDgG = tmem_copy_thread.partition_D(tCgG);
    Tensor tDgGGrouped = group_modes<1, decltype(rank(tDgG))::value>(tDgG);

    // Keep all S in registers, but only 16 transient product values at once.
    // Loading the full product alongside persistent S would spill registers.
    constexpr int kAccChunk = 16;
    CUTE_STATIC_ASSERT_V(size<1>(tDrSGrouped) % Int<kAccChunk>{} == Int<0>{});
#pragma unroll
    for (int chunk = 0; chunk < size<1>(tDrSGrouped) / kAccChunk; ++chunk) {
      Tensor tDtChunk = local_tile(tDtAccGrouped,
          make_shape(shape<0>(tDtAccGrouped), Int<kAccChunk>{}), make_coord(0, chunk));
      Tensor tDrChunk = make_tensor<AccType>(
          make_shape(shape<0>(tDrSGrouped), Int<kAccChunk>{}));
      copy(tmem_to_register, tDtChunk, tDrChunk);
      cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
      for (int i = 0; i < kAccChunk; ++i) {
        const int index = chunk * kAccChunk + i;
        tDrSGrouped(0, index) = tDrSGrouped(0, index) * tDgGGrouped(0, index) + tDrChunk(0, i);
      }
    }
    // Every warp must finish its TMEM read before warp zero can overwrite
    // that allocation with the next product. Also guards SMEM buffer reuse.
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
  }

  // The only state store: S_N. No recurrent state is written into TMEM.
  copy(tDrS, tDgC);

  __syncthreads();
  if (elected_warp) {
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
#else
  (void)kt;
  (void)u;
  (void)state;
  (void)g_total;
  (void)c;
  (void)steps;
  if (threadIdx.x == 0) asm volatile("trap;");
#endif
}

#include "p3_runner.cuh"

int main(int argc, char** argv) {
  auto launch = [](int batch, const BF16* kt, const BF16* u, const float* s,
                   const float* g, float* c, int steps) {
    tcgen05_p3_recurrence_kernel<<<batch, kThreads, kDynamicSmemBytes>>>(kt, u, s, g, c, steps);
  };
  return run_p3(argc, argv, "tcgen05_p3_recurrence", tcgen05_p3_recurrence_kernel, launch, kTmemColumns);
}
