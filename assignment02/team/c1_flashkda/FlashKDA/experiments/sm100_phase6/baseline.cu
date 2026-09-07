// P0 scaffold: K^T[128,16] @ U[16,128] -> C[128,128].
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
constexpr bool kImplemented = true;

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
__global__ void baseline_p0_kernel(const BF16* __restrict__ kt,
                                   const BF16* __restrict__ u,
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
      clear(tCrC);
      gemm(thr_mma, tCrA(_, _, Int<0>{}), tCrB(_, _, Int<0>{}), tCrC);
      copy(tCrC, tCgC);
    }
  }
}

struct Options {
  int warmup = 30;
  int iters = 200;
  int batch = 1024;
  unsigned seed = 2026;
  const char* input = "random";
  int probe_m = 0;
  int probe_k = 0;
  int probe_n = 0;
};

int parse_coordinate(const char* flag, const char* text, int extent) {
  char* end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0 || value >= extent) {
    std::fprintf(stderr, "invalid coordinate for %s: %s (extent %d)\n",
                 flag, text, extent);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

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
    const char* flag = argv[i];
    const char* value = argv[++i];
    if (std::strcmp(flag, "--warmup") == 0) {
      options.warmup = parse_positive(flag, value);
    } else if (std::strcmp(flag, "--iters") == 0) {
      options.iters = parse_positive(flag, value);
    } else if (std::strcmp(flag, "--batch") == 0) {
      options.batch = parse_positive(flag, value);
    } else if (std::strcmp(flag, "--seed") == 0) {
      options.seed = static_cast<unsigned>(parse_positive(flag, value));
    } else if (std::strcmp(flag, "--input") == 0) {
      options.input = value;
    } else if (std::strcmp(flag, "--m") == 0) {
      options.probe_m = parse_coordinate(flag, value, kM);
    } else if (std::strcmp(flag, "--k") == 0) {
      options.probe_k = parse_coordinate(flag, value, kK);
    } else if (std::strcmp(flag, "--n") == 0) {
      options.probe_n = parse_coordinate(flag, value, kN);
    } else {
      std::fprintf(stderr, "unknown option: %s\n", flag);
      std::exit(EXIT_FAILURE);
    }
  }
  if (std::strcmp(options.input, "random") != 0 &&
      std::strcmp(options.input, "one-hot") != 0 &&
      std::strcmp(options.input, "tagged") != 0) {
    std::fprintf(stderr, "--input must be random, one-hot, or tagged\n");
    std::exit(EXIT_FAILURE);
  }
  return options;
}

void initialize_inputs(const Options& options, std::vector<BF16>& kt,
                       std::vector<BF16>& u) {
  if (std::strcmp(options.input, "one-hot") == 0) {
    std::fill(kt.begin(), kt.end(), BF16(0.0f));
    std::fill(u.begin(), u.end(), BF16(0.0f));
    kt[options.probe_m * kK + options.probe_k] = BF16(1.0f);
    u[options.probe_k * kN + options.probe_n] = BF16(1.0f);
  } else if (std::strcmp(options.input, "tagged") == 0) {
    // Small dyadic values are exact in BF16; products and sums are exact in
    // FP32. Distinct row/column patterns exercise every reduction coordinate.
    for (int m = 0; m < kM; ++m)
      for (int k = 0; k < kK; ++k)
        kt[m * kK + k] = BF16(float((m * 7 + k * 3) % 31 - 15) / 16.0f);
    for (int k = 0; k < kK; ++k)
      for (int n = 0; n < kN; ++n)
        u[k * kN + n] = BF16(float((k * 11 + n * 5) % 29 - 14) / 16.0f);
  } else {
    std::mt19937 generator(options.seed);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    for (BF16& value : kt) value = BF16(distribution(generator));
    for (BF16& value : u) value = BF16(distribution(generator));
  }
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
  if (props.major != 10 || (props.minor != 0 && props.minor != 3)) {
    std::fprintf(stderr,
                 "baseline P0 requires Blackwell SM100/SM103; found "
                 "%d.%d (%s)\n",
                 props.major, props.minor, props.name);
    return 3;
  }

  std::vector<BF16> h_kt(kM * kK);
  std::vector<BF16> h_u(kK * kN);
  std::vector<float> h_ref(kM * kN);
  // Check both ends of the grid so a missing blockIdx.x output offset is caught.
  std::vector<float> h_got(2 * kM * kN);
  initialize_inputs(options, h_kt, h_u);
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
    baseline_p0_kernel<<<options.batch, kThreads, kDynamicSmemBytes>>>(d_kt, d_u, d_c);
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
  CUDA_CHECK(cudaGetLastError());
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
  const bool exact = std::strcmp(options.input, "random") != 0;
  bool reported_mismatch = false;
  for (size_t i = 0; i < h_got.size(); ++i) {
    finite = finite && std::isfinite(h_got[i]);
    const float reference = h_ref[i % h_ref.size()];
    const double abs_error = std::abs(static_cast<double>(h_got[i]) - reference);
    const double rel_error = abs_error / std::max(1.0e-6, std::abs(static_cast<double>(reference)));
    max_abs = std::max(max_abs, abs_error);
    max_rel = std::max(max_rel, rel_error);
    const double tolerance = exact ? 0.0 :
        1.0e-5 + 1.0e-5 * std::abs(static_cast<double>(reference));
    const bool matches = std::isfinite(h_got[i]) && abs_error <= tolerance;
    allclose = allclose && matches;
    if (!matches && !reported_mismatch) {
      const size_t index = i % h_ref.size();
      const int block = i < h_ref.size() ? 0 : options.batch - 1;
      std::fprintf(stderr,
          "first mismatch: block=%d m=%zu n=%zu expected=%.9g got=%.9g\n",
          block, index / kN, index % kN, reference, h_got[i]);
      reported_mismatch = true;
    }
  }

  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, baseline_p0_kernel));
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, baseline_p0_kernel, kThreads, kDynamicSmemBytes));

  const bool correct = kImplemented && finite && allclose;
  const double launch_us = elapsed_ms * 1000.0 / options.iters;
  const double cta_ns = launch_us * 1000.0 / options.batch;
  std::printf(
      "{\"implementation\":\"baseline\",\"implemented\":%s,"
      "\"input\":\"%s\",\"probe_m\":%d,\"probe_k\":%d,\"probe_n\":%d,"
      "\"device\":\"%s\",\"cc\":\"%d.%d\",\"correct\":%s,"
      "\"max_abs\":%.9g,\"max_rel\":%.9g,\"launch_us\":%.6f,"
      "\"cta_ns\":%.6f,\"static_smem_bytes\":%zu,"
      "\"dynamic_smem_bytes\":%d,\"registers_per_thread\":%d,"
      "\"blocks_per_sm\":%d,\"tmem_columns\":null}\n",
      kImplemented ? "true" : "false", options.input,
      options.probe_m, options.probe_k, options.probe_n,
      props.name, props.major, props.minor,
      correct ? "true" : "false", max_abs, max_rel, launch_us, cta_ns,
      attributes.sharedSizeBytes, kDynamicSmemBytes, attributes.numRegs,
      blocks_per_sm);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_kt));
  CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaFree(d_c));
  if (!kImplemented) return 2;
  return correct ? EXIT_SUCCESS : 4;
}
