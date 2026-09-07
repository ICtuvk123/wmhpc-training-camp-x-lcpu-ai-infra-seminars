// P1: FP32 state is seeded into TMEM before the BF16 MMA.
//
// This follows CUTLASS examples/cute/tutorial/blackwell/01_mma_sm100.cu, reduced
// to the exact Phase-6 P1 shape and a 128-column TMEM allocation.

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

// Layout contract matches baseline_p1.cu exactly.
__global__ void tcgen05_p1_kernel(const BF16* __restrict__ kt,
                                  const BF16* __restrict__ u,
                                  const float* __restrict__ state,
                                  float* __restrict__ c) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
  using namespace cute;

  auto tiled_mma = make_tcgen05_mma();
  auto mma_tiler = make_tcgen05_tiler();

  // tcgen05 represents B as [N,K].  The (1,N) stride is a transposed view of
  // the row-major U[K,N] allocation, so no materialized transpose is needed.
  Tensor m_a = make_tensor(
      make_gmem_ptr(kt),
      make_layout(make_shape(Int<kM>{}, Int<kK>{}),
                  make_stride(Int<kK>{}, Int<1>{})));
  Tensor m_b = make_tensor(
      make_gmem_ptr(u),
      make_layout(make_shape(Int<kN>{}, Int<kK>{}),
                  make_stride(Int<1>{}, Int<kN>{})));
  const float* block_s = state + static_cast<size_t>(blockIdx.x) * kM * kN;
  Tensor m_s = make_tensor(make_gmem_ptr(block_s),
      make_layout(make_shape(Int<kM>{}, Int<kN>{}),
                  make_stride(Int<kN>{}, Int<1>{})));
  float* block_c = c + static_cast<size_t>(blockIdx.x) * kM * kN;
  Tensor m_c = make_tensor(
      make_gmem_ptr(block_c),
      make_layout(make_shape(Int<kM>{}, Int<kN>{}),
                  make_stride(Int<kN>{}, Int<1>{})));

  auto mma_coord = make_coord(0, 0, _);
  Tensor g_a = local_tile(m_a, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor g_b = local_tile(m_b, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor g_s = local_tile(m_s, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor g_c = local_tile(m_c, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto& storage = *reinterpret_cast<Tcgen05SharedStorage*>(smem_raw);
  Tensor s_a = storage.tensor_a();
  Tensor s_b = storage.tensor_b();

  ThrMMA cta_mma = tiled_mma.get_slice(Int<0>{});
  Tensor tCgA = cta_mma.partition_A(g_a);
  Tensor tCgB = cta_mma.partition_B(g_b);
  Tensor tCgS = cta_mma.partition_C(g_s);
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
  }
  __syncthreads();

  cooperative_copy<kThreads>(threadIdx.x, tCgA(_, _, _, Int<0>{}), s_a);
  cooperative_copy<kThreads>(threadIdx.x, tCgB(_, _, _, Int<0>{}), s_b);
  __syncthreads();

  // GMEM -> small register fragment -> TMEM, never a full SMEM state tile.
  // Partition S using the STORE atom's layout, so its coordinates agree with
  // the actual MMA accumulator (rather than assuming lane == matrix row).
  {
    auto store = make_tmem_copy(SM100_TMEM_STORE_32dp32b1x{}, tCtAcc);
    auto thread_store = store.get_slice(threadIdx.x);
    Tensor source = thread_store.partition_S(tCgS);
    Tensor destination = thread_store.partition_D(tCtAcc);
    Tensor source_groups = group_modes<1, rank(source)>(source);
    Tensor destination_groups = group_modes<1, rank(destination)>(destination);
    Tensor fragment = make_tensor<float>(shape(source_groups(_, 0)));
    // Keep only one copy atom's values live. Disable unrolling so this does
    // not materialize a second full 128-float/thread state fragment.
#pragma unroll 1
    for (int i = 0; i < size<1>(source_groups); ++i) {
      copy(source_groups(_, i), fragment);
      copy(store, fragment, destination_groups(_, i));
    }
    cutlass::arch::fence_view_async_tmem_store();  // tcgen05.wait::st
  }
  // Publish ordinary SMEM operand writes to the async MMA proxy, and order
  // every warp's TMEM stores before the issuing warp's MMA.
  cutlass::arch::fence_view_async_shared();
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  tiled_mma.accumulate_ = UMMA::ScaleOut::One;
  if (elected_warp) {
#pragma unroll
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
      tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
    cutlass::arch::umma_arrive(&storage.mma_barrier);
  }
  cute::wait_barrier(storage.mma_barrier, 0);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  TiledCopy tmem_to_register =
      make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy tmem_copy_thread = tmem_to_register.get_slice(threadIdx.x);
  Tensor tDtAcc = tmem_copy_thread.partition_S(tCtAcc);
  Tensor tDgC = tmem_copy_thread.partition_D(tCgC);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrAcc = make_tensor<AccType>(shape(tDgC));
  copy(tmem_to_register, tDtAcc, tDrAcc);
  cutlass::arch::fence_view_async_tmem_load();  // tcgen05.wait::ld
  copy(tDrAcc, tDgC);
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

  __syncthreads();
  if (elected_warp) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
#else
  (void)kt;
  (void)u;
  (void)c;
  (void)state;
  if (threadIdx.x == 0) asm volatile("trap;");
#endif
}

#include "p1_runner.cuh"

int main(int argc, char** argv) {
  auto launch = [](int batch, const BF16* kt, const BF16* u, const float* s, float* c) {
    tcgen05_p1_kernel<<<batch, kThreads, kDynamicSmemBytes>>>(kt, u, s, c);
  };
  return run_p1(argc, argv, "tcgen05", tcgen05_p1_kernel, launch, kTmemColumns);
}
