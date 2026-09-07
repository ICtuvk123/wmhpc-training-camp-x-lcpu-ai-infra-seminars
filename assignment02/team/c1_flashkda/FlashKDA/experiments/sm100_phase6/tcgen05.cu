// P0 scaffold: tcgen05/TMEM version of [128,16] @ [16,128].
//
// This follows CUTLASS examples/cute/tutorial/blackwell/01_mma_sm100.cu, reduced
// to the exact Phase-6 P0 shape and a 128-column TMEM allocation.

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
constexpr bool kImplemented = true;

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

// Layout contract matches baseline.cu exactly.
__global__ void tcgen05_p0_kernel(const BF16* __restrict__ kt,
                                  const BF16* __restrict__ u,
                                  float* __restrict__ c) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
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
  float* block_c = c + static_cast<size_t>(blockIdx.x) * kM * kN;
  Tensor m_c = make_tensor(
      make_gmem_ptr(block_c),
      make_layout(make_shape(Int<kM>{}, Int<kN>{}),
                  make_stride(Int<kN>{}, Int<1>{})));

  auto mma_coord = make_coord(0, 0, _);
  Tensor g_a = local_tile(m_a, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor g_b = local_tile(m_b, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor g_c = local_tile(m_c, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto& storage = *reinterpret_cast<Tcgen05SharedStorage*>(smem_raw);
  Tensor s_a = storage.tensor_a();
  Tensor s_b = storage.tensor_b();

  ThrMMA cta_mma = tiled_mma.get_slice(Int<0>{});
  Tensor tCgA = cta_mma.partition_A(g_a);
  Tensor tCgB = cta_mma.partition_B(g_b);
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
  tCtAcc.data() = storage.tmem_base_ptr;

  if (elected_warp && elected_thread) {
    cute::initialize_barrier(storage.mma_barrier, 1);
  }
  __syncthreads();

  cooperative_copy<kThreads>(threadIdx.x, tCgA(_, _, _, Int<0>{}), s_a);
  cooperative_copy<kThreads>(threadIdx.x, tCgB(_, _, _, Int<0>{}), s_b);
  __syncthreads();

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  if (elected_warp) {
#pragma unroll
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
      tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
    cutlass::arch::umma_arrive(&storage.mma_barrier);
  }
  cute::wait_barrier(storage.mma_barrier, 0);

  TiledCopy tmem_to_register =
      make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy tmem_copy_thread = tmem_to_register.get_slice(threadIdx.x);
  Tensor tDtAcc = tmem_copy_thread.partition_S(tCtAcc);
  Tensor tDgC = tmem_copy_thread.partition_D(tCgC);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrAcc = make_tensor<AccType>(shape(tDgC));
  copy(tmem_to_register, tDtAcc, tDrAcc);
  copy(tDrAcc, tDgC);

  __syncthreads();
  if (elected_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(storage.tmem_base_ptr, kTmemColumns);
  }
#else
  (void)kt;
  (void)u;
  (void)c;
  if (threadIdx.x == 0) asm volatile("trap;");
#endif
}

struct Options {
  int warmup = 30;
  int iters = 200;
  int batch = 1024;
  unsigned seed = 2026;
};

int parse_positive(const char* flag, const char* text) {
  char* end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0 || value > (1L << 30)) {
    std::fprintf(stderr, "invalid value for %s: %s\n", flag, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "missing value after %s\n", argv[i]);
      std::exit(EXIT_FAILURE);
    }
    if (std::strcmp(argv[i], "--warmup") == 0) {
      options.warmup = parse_positive(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      options.iters = parse_positive(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--batch") == 0) {
      options.batch = parse_positive(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--seed") == 0) {
      options.seed = static_cast<unsigned>(parse_positive(argv[i], argv[++i]));
    } else {
      std::fprintf(stderr, "unknown option: %s\n", argv[i]);
      std::exit(EXIT_FAILURE);
    }
  }
  return options;
}

void reference_gemm(const std::vector<BF16>& kt, const std::vector<BF16>& u,
                    std::vector<float>& c) {
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < kK; ++k) {
        acc += static_cast<float>(kt[m * kK + k]) *
               static_cast<float>(u[k * kN + n]);
      }
      c[m * kN + n] = acc;
    }
  }
}

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);

  int device = 0;
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  if (props.major != 10 || props.minor != 0) {
    std::fprintf(stderr, "tcgen05 P0 requires an SM100 device; found %d.%d (%s)\n",
                 props.major, props.minor, props.name);
    return 3;
  }

  std::vector<BF16> h_kt(kM * kK);
  std::vector<BF16> h_u(kK * kN);
  std::vector<float> h_ref(kM * kN);
  // Check both ends of the grid so a missing blockIdx.x output offset is caught.
  std::vector<float> h_got(2 * kM * kN);
  std::mt19937 generator(options.seed);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  for (BF16& value : h_kt) value = BF16(distribution(generator));
  for (BF16& value : h_u) value = BF16(distribution(generator));
  reference_gemm(h_kt, h_u, h_ref);

  BF16* d_kt = nullptr;
  BF16* d_u = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_kt, h_kt.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_u, h_u.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(options.batch) * kM * kN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_kt, h_kt.data(), h_kt.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), h_u.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0xff,
                       static_cast<size_t>(options.batch) * kM * kN * sizeof(float)));

  auto launch = [&] {
    tcgen05_p0_kernel<<<options.batch, kThreads, kDynamicSmemBytes>>>(d_kt, d_u, d_c);
  };
  for (int i = 0; i < options.warmup; ++i) launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < options.iters; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaMemcpy(h_got.data(), d_c, kM * kN * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_got.data() + kM * kN,
                        d_c + static_cast<size_t>(options.batch - 1) * kM * kN,
                        kM * kN * sizeof(float), cudaMemcpyDeviceToHost));
  double max_abs = 0.0;
  double max_rel = 0.0;
  bool finite = true;
  bool allclose = true;
  for (size_t i = 0; i < h_ref.size(); ++i) {
    finite = finite && std::isfinite(h_got[i]);
    const float reference = h_ref[i % h_ref.size()];
    const double abs_error = std::abs(static_cast<double>(h_got[i]) - reference);
    const double rel_error = abs_error / std::max(1.0e-6, std::abs(static_cast<double>(reference)));
    max_abs = std::max(max_abs, abs_error);
    max_rel = std::max(max_rel, rel_error);
    allclose = allclose && abs_error <= 2.0e-2 + 2.0e-2 * std::abs(static_cast<double>(reference));
  }

  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, tcgen05_p0_kernel));
  int runtime_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &runtime_blocks_per_sm, tcgen05_p0_kernel, kThreads, kDynamicSmemBytes));
  constexpr int kTmemCapacityColumns = cute::TMEM::Sm100TmemCapacityColumns;
  const int tmem_blocks_per_sm = kTmemCapacityColumns / kTmemColumns;
  const int blocks_per_sm = std::min(runtime_blocks_per_sm, tmem_blocks_per_sm);

  const bool correct = kImplemented && finite && allclose;
  const double launch_us = elapsed_ms * 1000.0 / options.iters;
  const double cta_ns = launch_us * 1000.0 / options.batch;
  std::printf(
      "{\"implementation\":\"tcgen05\",\"implemented\":%s,"
      "\"device\":\"%s\",\"cc\":\"%d.%d\",\"correct\":%s,"
      "\"max_abs\":%.9g,\"max_rel\":%.9g,\"launch_us\":%.6f,"
      "\"cta_ns\":%.6f,\"static_smem_bytes\":%zu,"
      "\"dynamic_smem_bytes\":%d,\"registers_per_thread\":%d,"
      "\"runtime_blocks_per_sm\":%d,\"tmem_blocks_per_sm\":%d,"
      "\"blocks_per_sm\":%d,\"tmem_columns\":%d}\n",
      kImplemented ? "true" : "false", props.name, props.major, props.minor,
      correct ? "true" : "false", max_abs, max_rel, launch_us, cta_ns,
      attributes.sharedSizeBytes, kDynamicSmemBytes, attributes.numRegs,
      runtime_blocks_per_sm, tmem_blocks_per_sm, blocks_per_sm, kTmemColumns);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_kt));
  CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaFree(d_c));
  if (!kImplemented) return 2;
  return correct ? EXIT_SUCCESS : 4;
}
