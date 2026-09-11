// Compare the real production K2 baseline and V0 in one process. Capture each
// full forward, retain its terminal K2 node, and time that node without K1 or
// a profiler. Full K1 is executed once to prepare the shared real workspace.
#include "../../csrc/fwd.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using BF16 = cutlass::bfloat16_t;
#define CHECK(expr) do { auto status = (expr); if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA status=%d: %s\n", __FILE__, __LINE__, int(status), cudaGetErrorString(status)); \
    std::exit(1); } } while (0)

template <class T> struct Buffer {
    T* ptr = nullptr;
    size_t size;
    explicit Buffer(size_t n): size(n) { CHECK(cudaMalloc(&ptr, n * sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
    void upload(const std::vector<T>& values) {
        if (values.size() != size) std::abort();
        CHECK(cudaMemcpy(ptr, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> values(size);
        CHECK(cudaMemcpy(values.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

struct K2Graph {
    cudaGraph_t full = nullptr, k2 = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaFuncAttributes attributes{};
    size_t dynamic_smem = 0;
    void extract() {
        size_t count = 0;
        CHECK(cudaGraphGetNodes(full, nullptr, &count));
        // Fixed-length launch_fwd must contain exactly K1 -> K2. Fail loudly
        // if production launch structure changes; never silently time K1.
        if (count != 2) { std::fprintf(stderr, "expected two forward graph nodes, got %zu\n", count); std::exit(2); }
        std::vector<cudaGraphNode_t> nodes(count);
        CHECK(cudaGraphGetNodes(full, nodes.data(), &count));
        cudaGraphNode_t leaf = nullptr;
        for (auto node : nodes) {
            cudaGraphNodeType type;
            CHECK(cudaGraphNodeGetType(node, &type));
            if (type != cudaGraphNodeTypeKernel) { std::fprintf(stderr, "expected kernel graph nodes\n"); std::exit(2); }
            size_t children = 0;
            CHECK(cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &children));
            if (children == 0) {
                if (leaf != nullptr) std::abort();
                leaf = node;
            }
        }
        if (!leaf) std::abort();
        cudaKernelNodeParams params{};
        CHECK(cudaGraphKernelNodeGetParams(leaf, &params));
        if (params.blockDim.x != 192) { std::fprintf(stderr, "terminal node is not 192-thread K2\n"); std::exit(2); }
        CHECK(cudaFuncGetAttributes(&attributes, params.func));
        dynamic_smem = params.sharedMemBytes;
        CHECK(cudaGraphCreate(&k2, 0));
        cudaGraphNode_t copied;
        CHECK(cudaGraphAddKernelNode(&copied, k2, nullptr, 0, &params));
        CHECK(cudaGraphInstantiate(&exec, k2, nullptr, nullptr, 0));
    }
    ~K2Graph() {
        if (exec) cudaGraphExecDestroy(exec);
        if (k2) cudaGraphDestroy(k2);
        if (full) cudaGraphDestroy(full);
    }
};

static bool equal(const std::vector<BF16>& expected, const std::vector<BF16>& actual, const char* label) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(float(expected[i])) || !std::isfinite(float(actual[i])) ||
            expected[i].storage != actual[i].storage) {
            std::fprintf(stderr, "%s first mismatch at %zu: baseline=%.9g V0=%.9g\n",
                         label, i, float(expected[i]), float(actual[i]));
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int batch = 4, tokens = 2048, warmup = 30, iters = 200, rounds = 5;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) return 2;
        char* end = nullptr;
        long n = std::strtol(argv[i + 1], &end, 10);
        if (end == argv[i + 1] || *end || n < 1 || n > 8192) return 2;
        if (!std::strcmp(argv[i], "--batch")) batch = int(n);
        else if (!std::strcmp(argv[i], "--tokens")) tokens = int(n);
        else if (!std::strcmp(argv[i], "--warmup")) warmup = int(n);
        else if (!std::strcmp(argv[i], "--iters")) iters = int(n);
        else if (!std::strcmp(argv[i], "--rounds")) rounds = int(n);
        else return 2;
    }
    if (!((batch == 4 && tokens == 2048) || (batch == 1 && tokens == 8192) ||
          (batch == 8 && tokens == 1024))) {
        std::fprintf(stderr, "Use frozen (B,T): (4,2048), (1,8192), or (8,1024)\n"); return 2;
    }
    constexpr int H = 64, D = 128, chunk = 16;
    int device; cudaDeviceProp props{};
    CHECK(cudaGetDevice(&device)); CHECK(cudaGetDeviceProperties(&props, device));
    if (props.major != 10 || (props.minor != 0 && props.minor != 3)) {
        std::fprintf(stderr, "Requires SM100/SM103, found %d.%d\n", props.major, props.minor); return 3;
    }
    const int total = batch * tokens, tiles = batch * (tokens / chunk);
    const size_t elements = size_t(total) * H * D, state_elements = size_t(batch) * H * D * D;
    Buffer<BF16> q(elements), k(elements), v(elements), g(elements), beta(size_t(total) * H);
    Buffer<BF16> initial(state_elements), out_base(elements), out_v0(elements), final_base(state_elements), final_v0(state_elements);
    Buffer<float> alog(H), bias(H * D);
    const size_t per_tile = 3 * chunk * D * sizeof(BF16) + D * sizeof(float) + 2 * chunk * chunk * sizeof(BF16);
    const size_t prefix = ((batch + 1) * sizeof(int) + 127) / 128 * 128;
    Buffer<unsigned char> workspace(size_t(H) * tiles * per_tile + prefix);
    std::mt19937 rng(42);
    std::normal_distribution<float> random(0.f, 1.f);
    std::vector<BF16> values(elements);
    for (auto* input : {&q, &k, &v, &g}) {
        for (auto& x : values) x = BF16(random(rng));
        input->upload(values);
    }
    values.resize(beta.size);
    for (auto& x : values) x = BF16(random(rng));
    // launch_fwd expects beta [H,T_total]; random IID values already fit.
    beta.upload(values);
    values.resize(initial.size);
    for (size_t i = 0; i < values.size(); ++i) values[i] = BF16(float(int(i % 127) - 63) / 128.f);
    initial.upload(values);
    alog.upload(std::vector<float>(H, 0.f));
    bias.upload(std::vector<float>(H * D, 0.25f));
    CHECK(cudaMemset(out_base.ptr, 0xff, elements * sizeof(BF16)));
    CHECK(cudaMemset(out_v0.ptr, 0xff, elements * sizeof(BF16)));
    CHECK(cudaMemset(final_base.ptr, 0xff, state_elements * sizeof(BF16)));
    CHECK(cudaMemset(final_v0.ptr, 0xff, state_elements * sizeof(BF16)));
    cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
    auto baseline = [&]() { launch_fwd<128, true, true, false, false, false>(
        q.ptr, k.ptr, v.ptr, g.ptr, beta.ptr, initial.ptr, 1.f / std::sqrt(float(D)),
        final_base.ptr, out_base.ptr, workspace.ptr, tiles, total, H, batch, nullptr,
        alog.ptr, bias.ptr, -5.f * 1.4426950408889634f, stream); };
    auto v0 = [&]() { launch_fwd<128, true, true, false, false, true>(
        q.ptr, k.ptr, v.ptr, g.ptr, beta.ptr, initial.ptr, 1.f / std::sqrt(float(D)),
        final_v0.ptr, out_v0.ptr, workspace.ptr, tiles, total, H, batch, nullptr,
        alog.ptr, bias.ptr, -5.f * 1.4426950408889634f, stream); };
    baseline(); CHECK(cudaGetLastError()); CHECK(cudaStreamSynchronize(stream)); // real K1 workspace
    K2Graph graphs[2];
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)); baseline();
    CHECK(cudaStreamEndCapture(stream, &graphs[0].full)); graphs[0].extract();
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)); v0();
    CHECK(cudaStreamEndCapture(stream, &graphs[1].full)); graphs[1].extract();
    for (auto& graph : graphs) CHECK(cudaGraphLaunch(graph.exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    const bool correct = equal(out_base.download(), out_v0.download(), "output") &&
                         equal(final_base.download(), final_v0.download(), "final_state");
    if (!correct) { std::puts("{\"correct\":false}"); return 4; }
    cudaEvent_t start, stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    double sums[2] = {0, 0};
    for (int round = 0; round < rounds; ++round) for (int order = 0; order < 2; ++order) {
        const int impl = round % 2 ? 1 - order : order;
        auto& graph = graphs[impl];
        for (int i = 0; i < warmup; ++i) CHECK(cudaGraphLaunch(graph.exec, stream));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < iters; ++i) CHECK(cudaGraphLaunch(graph.exec, stream));
        CHECK(cudaEventRecord(stop, stream)); CHECK(cudaEventSynchronize(stop));
        float ms; CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double us = ms * 1000.0 / iters;
        sums[impl] += us;
        std::printf("{\"implementation\":\"%s\",\"correct\":true,\"B\":%d,\"T\":%d,\"H\":64,\"D\":128,"
                    "\"round\":%d,\"k2_us\":%.6f,\"registers_per_thread\":%d,\"dynamic_smem_bytes\":%zu,"
                    "\"local_bytes_per_thread\":%zu,\"timing\":\"single_K2_graph_node\"}\n",
                    impl ? "sm100_v0" : "baseline", batch, tokens, round + 1, us,
                    graph.attributes.numRegs, graph.dynamic_smem, graph.attributes.localSizeBytes);
        std::fflush(stdout);
    }
    if (!equal(out_base.download(), out_v0.download(), "timed output") ||
        !equal(final_base.download(), final_v0.download(), "timed final_state")) return 4;
    std::printf("{\"summary\":true,\"baseline_k2_us\":%.6f,\"sm100_v0_k2_us\":%.6f,\"speedup\":%.6f}\n",
                sums[0] / rounds, sums[1] / rounds, sums[0] / sums[1]);
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop)); CHECK(cudaStreamDestroy(stream));
}
