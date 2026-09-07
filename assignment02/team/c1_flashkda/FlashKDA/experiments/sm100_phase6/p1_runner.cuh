// Shared host contract for the two isolated P1 implementations.
#pragma once

#include <limits>
#include <string>

struct P1Options {
  int warmup = 30;
  int iters = 200;
  int batch = 1024;
  unsigned seed = 2026;
  std::string input = "random";
  std::string state = "tagged";
  int m = 5;
  int k = 3;
  int n = 37;
};

inline int p1_integer(const char* flag, const char* value, int minimum, int maximum) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end || parsed < minimum || parsed > maximum) {
    std::fprintf(stderr, "invalid %s: %s\n", flag, value);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(parsed);
}

inline P1Options p1_options(int argc, char** argv) {
  P1Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (i + 1 == argc) {
      std::fprintf(stderr, "missing value after %s\n", flag.c_str());
      std::exit(EXIT_FAILURE);
    }
    const char* value = argv[++i];
    if (flag == "--input") o.input = value;
    else if (flag == "--state") o.state = value;
    else if (flag == "--m") o.m = p1_integer(flag.c_str(), value, 0, kM - 1);
    else if (flag == "--k") o.k = p1_integer(flag.c_str(), value, 0, kK - 1);
    else if (flag == "--n") o.n = p1_integer(flag.c_str(), value, 0, kN - 1);
    else if (flag == "--warmup") o.warmup = p1_integer(flag.c_str(), value, 1, 1 << 30);
    else if (flag == "--iters") o.iters = p1_integer(flag.c_str(), value, 1, 1 << 30);
    else if (flag == "--batch") o.batch = p1_integer(flag.c_str(), value, 1, 1 << 20);
    else if (flag == "--seed") o.seed = p1_integer(flag.c_str(), value, 1, 1 << 30);
    else {
      std::fprintf(stderr, "unknown option: %s\n", flag.c_str());
      std::exit(EXIT_FAILURE);
    }
  }
  if ((o.input != "random" && o.input != "zero" && o.input != "one-hot" && o.input != "tagged") ||
      (o.state != "zero" && o.state != "tagged" && o.state != "random")) {
    std::fprintf(stderr, "--input: random/zero/one-hot/tagged; --state: zero/tagged/random\n");
    std::exit(EXIT_FAILURE);
  }
  return o;
}

inline void p1_inputs(const P1Options& o, std::vector<BF16>& kt,
                      std::vector<BF16>& u, std::vector<float>& state) {
  std::mt19937 generator(o.seed);
  std::uniform_real_distribution<float> random(-0.5f, 0.5f);
  std::fill(kt.begin(), kt.end(), BF16(0.0f));
  std::fill(u.begin(), u.end(), BF16(0.0f));
  if (o.input == "one-hot") {
    kt[o.m * kK + o.k] = BF16(1.0f);
    u[o.k * kN + o.n] = BF16(1.0f);
  } else if (o.input == "tagged") {
    for (int m = 0; m < kM; ++m)
      for (int k = 0; k < kK; ++k)
        kt[m * kK + k] = BF16(float((m * 7 + k * 3) % 31 - 15) / 16.0f);
    for (int k = 0; k < kK; ++k)
      for (int n = 0; n < kN; ++n)
        u[k * kN + n] = BF16(float((k * 11 + n * 5) % 29 - 14) / 16.0f);
  } else if (o.input == "random") {
    for (auto& x : kt) x = BF16(random(generator));
    for (auto& x : u) x = BF16(random(generator));
  }
  for (size_t i = 0; i < state.size(); ++i) {
    // Binary fractions tag every row/column and distinguish CTAs. The fine
    // fraction also catches an accidental FP32 -> BF16 state conversion.
    const int block = static_cast<int>(i / (kM * kN));
    const int element = static_cast<int>(i % (kM * kN));
    state[i] = o.state == "zero" ? 0.0f : o.state == "random" ? random(generator) :
        float(element - 8192) / 16384.0f + float(block % 127 + 1) / 16.0f;
  }
}

template <class Kernel, class Launch>
int run_p1(int argc, char** argv, const char* implementation, Kernel kernel,
           Launch launch, int tmem_columns) {
  const P1Options o = p1_options(argc, argv);
  int device = 0;
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  if (props.major != 10 || (props.minor != 0 && props.minor != 3)) {
    std::fprintf(stderr, "P1 requires Blackwell SM100/SM103; found %d.%d (%s)\n",
                 props.major, props.minor, props.name);
    return 3;
  }

  const size_t elements = static_cast<size_t>(o.batch) * kM * kN;
  const size_t state_bytes = elements * sizeof(float);
  std::vector<BF16> kt(kM * kK), u(kK * kN);
  std::vector<float> state(elements), got(elements);
  p1_inputs(o, kt, u, state);
  // Independent, higher-precision oracle from the actual rounded BF16 inputs.
  std::vector<double> product(kM * kN, 0.0);
  for (int m = 0; m < kM; ++m)
    for (int n = 0; n < kN; ++n)
      for (int k = 0; k < kK; ++k)
        product[m * kN + n] += double(float(kt[m * kK + k])) * double(float(u[k * kN + n]));

  BF16 *d_kt = nullptr, *d_u = nullptr;
  float *d_s = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_kt, kt.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_u, u.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_s, state_bytes));
  CUDA_CHECK(cudaMalloc(&d_c, state_bytes));
  CUDA_CHECK(cudaMemcpy(d_kt, kt.data(), kt.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_u, u.data(), u.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_s, state.data(), state_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0xff, state_bytes));

  // S is immutable and separate from C: every warmup/timed launch is ONE P1
  // update, never an accidental recurrence across timing iterations.
  for (int i = 0; i < o.warmup; ++i) launch(o.batch, d_kt, d_u, d_s, d_c);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < o.iters; ++i) launch(o.batch, d_kt, d_u, d_s, d_c);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaMemcpy(got.data(), d_c, state_bytes, cudaMemcpyDeviceToHost));

  // Deterministic dyadic inputs require exact results; random cases allow
  // FP32 accumulation ordering differences, with no BF16-sized tolerance.
  const bool exact = o.input == "zero" || (o.input != "random" && o.state != "random");
  bool correct = true;
  double max_abs = 0.0, max_rel = 0.0;
  for (size_t i = 0; i < elements; ++i) {
    const float ref = static_cast<float>(double(state[i]) + product[i % (kM * kN)]);
    const double error = std::abs(double(got[i]) - ref);
    const double tolerance = exact ? 0.0 : 1.0e-5 + 1.0e-5 * std::abs(double(ref));
    const bool match = std::isfinite(got[i]) && error <= tolerance;
    if (!match && correct)
      std::fprintf(stderr, "first mismatch: block=%zu m=%zu n=%zu expected=%.9g got=%.9g\n",
          i / (kM * kN), (i / kN) % kM, i % kN, ref, got[i]);
    correct = correct && match;
    if (std::isfinite(error)) {
      max_abs = std::max(max_abs, error);
      max_rel = std::max(max_rel, error / std::max(1.0e-6, std::abs(double(ref))));
    } else {
      max_abs = max_rel = std::numeric_limits<double>::max();
    }
  }
  CUDA_CHECK(cudaMemcpy(got.data(), d_s, state_bytes, cudaMemcpyDeviceToHost));
  const bool state_unchanged = std::memcmp(got.data(), state.data(), state_bytes) == 0;
  correct = correct && state_unchanged;

  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int runtime_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &runtime_blocks, kernel, kThreads, kDynamicSmemBytes));
  // This is a capacity estimate, NOT measured residency/achieved occupancy.
  const int blocks = tmem_columns ? std::min(runtime_blocks, 512 / tmem_columns) : runtime_blocks;
  const double launch_us = elapsed_ms * 1000.0 / o.iters;
  std::printf(
      "{\"stage\":\"p1\",\"operation\":\"S+K^T U\",\"implementation\":\"%s\","
      "\"implemented\":true,\"device\":\"%s\",\"cc\":\"%d.%d\",\"correct\":%s,"
      "\"state_unchanged\":%s,\"input\":\"%s\",\"state\":\"%s\",\"exact\":%s,"
      "\"probe_m\":%d,\"probe_k\":%d,\"probe_n\":%d,\"batch\":%d,\"seed\":%u,"
      "\"warmup\":%d,\"iters\":%d,\"checked_ctas\":%d,"
      "\"max_abs\":%.9g,\"max_rel\":%.9g,\"launch_us\":%.6f,\"cta_ns\":%.6f,"
      "\"static_smem_bytes\":%zu,\"dynamic_smem_bytes\":%d,\"registers_per_thread\":%d,"
      "\"local_bytes_per_thread\":%zu,\"runtime_blocks_per_sm\":%d,\"blocks_per_sm\":%d,"
      "\"tmem_columns\":%d,\"achieved_occupancy_pct\":null}\n",
      implementation, props.name, props.major, props.minor, correct ? "true" : "false",
      state_unchanged ? "true" : "false", o.input.c_str(), o.state.c_str(), exact ? "true" : "false",
      o.m, o.k, o.n, o.batch, o.seed, o.warmup, o.iters, o.batch,
      max_abs, max_rel, launch_us, launch_us * 1000.0 / o.batch,
      attributes.sharedSizeBytes, kDynamicSmemBytes, attributes.numRegs,
      attributes.localSizeBytes, runtime_blocks, blocks, tmem_columns);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_kt));
  CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaFree(d_s));
  CUDA_CHECK(cudaFree(d_c));
  return correct ? EXIT_SUCCESS : 4;
}
