// Isolated V1 feasibility probe, not a production K2 implementation.
// Load BF16 state in production's SM80 Phase-6 C distribution, MOVM_T it to
// Phase-1 B fragments, retain all state in registers, and compute kS/qS for
// multiple changing k/q tiles. No shared-memory state exists in this kernel.
#include "../../csrc/smxx/utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

using BF16 = cutlass::bfloat16_t;
constexpr int D = 128, CHUNK = 16;

#define CHECK_CUDA(expr) do { auto status = (expr); if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA status=%d: %s\n", __FILE__, __LINE__, int(status), \
                 cudaGetErrorString(status)); std::exit(1); } } while (0)

__global__ void phase1_register_state_kernel(const BF16* state, const BF16* k,
    const BF16* q, float* out_k, float* out_q, BF16* state_roundtrip, int steps) {
    using namespace cute;
    auto mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                              Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
    const int warp = threadIdx.x / 32;
    auto thr = mma.get_slice(threadIdx.x % 32);
    auto tile_layout = make_layout(make_shape(_16{}, _16{}), make_stride(_16{}, _1{}));
    Tensor ref = make_tensor(make_gmem_ptr(k), tile_layout);
    using BFrag = decltype(thr.partition_fragment_B(ref));
    BFrag state_b[2][D / CHUNK];

    // State storage is [value,key]. Phase-6 C views it as [key,value].
    Tensor state_t = make_tensor(make_gmem_ptr(state + size_t(blockIdx.x) * D * D),
        make_layout(make_shape(Int<D>{}, Int<D>{}), make_stride(_1{}, Int<D>{})));
#pragma unroll
    for (int bi = 0; bi < 2; ++bi) {
#pragma unroll
        for (int kb = 0; kb < D / CHUNK; ++kb) {
            Tensor tile = local_tile(state_t, make_shape(_16{}, _16{}), make_coord(kb, warp * 2 + bi));
            Tensor state_c = make_fragment_like<BF16>(thr.partition_C(tile));
            copy(thr.partition_C(tile), state_c);
            state_b[bi][kb] = thr.partition_fragment_B(ref);
            CUTE_STATIC_ASSERT_V(size(state_c) == Int<8>{});
            CUTE_STATIC_ASSERT_V(size(state_b[bi][kb]) == Int<8>{});
            auto* src = reinterpret_cast<uint32_t*>(&state_c(0));
            auto* dst = reinterpret_cast<uint32_t*>(&state_b[bi][kb](0));
            // Same C-to-B transform as production's U conversion in Phase 4.
#pragma unroll
            for (int word = 0; word < 4; ++word) SM75_U32x1_MOVM_T::copy(src[word], dst[word]);
        }
    }

#pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        Tensor k_tile = make_tensor(make_gmem_ptr(k + size_t(step) * CHUNK * D),
            make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), make_stride(Int<D>{}, _1{})));
        Tensor q_tile = make_tensor(make_gmem_ptr(q + size_t(step) * CHUNK * D), k_tile.layout());
        const size_t out_offset = (size_t(blockIdx.x) * steps + step) * CHUNK * D;
        Tensor yk = make_tensor(make_gmem_ptr(out_k + out_offset), k_tile.layout());
        Tensor yq = make_tensor(make_gmem_ptr(out_q + out_offset), k_tile.layout());
#pragma unroll
        for (int bi = 0; bi < 2; ++bi) {
            Tensor g_yk = local_tile(yk, make_shape(_16{}, _16{}), make_coord(0, warp * 2 + bi));
            Tensor g_yq = local_tile(yq, make_shape(_16{}, _16{}), make_coord(0, warp * 2 + bi));
            Tensor acc_k = thr.make_fragment_C(thr.partition_C(g_yk));
            Tensor acc_q = thr.make_fragment_C(thr.partition_C(g_yq));
            clear(acc_k); clear(acc_q);
#pragma unroll
            for (int kb = 0; kb < D / CHUNK; ++kb) {
                Tensor g_k = local_tile(k_tile, make_shape(_16{}, _16{}), make_coord(0, kb));
                Tensor g_q = local_tile(q_tile, make_shape(_16{}, _16{}), make_coord(0, kb));
                Tensor a_k = thr.partition_fragment_A(g_k);
                Tensor a_q = thr.partition_fragment_A(g_q);
                copy(thr.partition_A(g_k), a_k);
                copy(thr.partition_A(g_q), a_q);
                gemm(thr, a_k(_, _, Int<0>{}), state_b[bi][kb](_, _, Int<0>{}), acc_k);
                gemm(thr, a_q(_, _, Int<0>{}), state_b[bi][kb](_, _, Int<0>{}), acc_q);
            }
            copy(acc_k, thr.partition_C(g_yk));
            copy(acc_q, thr.partition_C(g_yq));
        }
    }

    // Independently expose the register B representation in [value,key]
    // order so the test catches a permutation even before checking GEMMs.
    Tensor restored = make_tensor(make_gmem_ptr(state_roundtrip + size_t(blockIdx.x) * D * D),
        make_layout(make_shape(Int<D>{}, Int<D>{}), make_stride(Int<D>{}, _1{})));
#pragma unroll
    for (int bi = 0; bi < 2; ++bi) {
#pragma unroll
        for (int kb = 0; kb < D / CHUNK; ++kb) {
            Tensor tile = local_tile(restored, make_shape(_16{}, _16{}), make_coord(warp * 2 + bi, kb));
            copy(state_b[bi][kb], thr.partition_B(tile));
        }
    }
}

int main(int argc, char** argv) {
    int steps = 4, batch = 3, warmup = 1, iters = 1;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) return 2;
        char* end = nullptr;
        const long value = std::strtol(argv[i + 1], &end, 10);
        if (end == argv[i + 1] || *end || value < 1 || value > 1024) return 2;
        if (!std::strcmp(argv[i], "--steps") && value <= 16) steps = int(value);
        else if (!std::strcmp(argv[i], "--batch")) batch = int(value);
        else if (!std::strcmp(argv[i], "--warmup")) warmup = int(value);
        else if (!std::strcmp(argv[i], "--iters")) iters = int(value);
        else return 2;
    }
    int device; cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDevice(&device)); CHECK_CUDA(cudaGetDeviceProperties(&props, device));
    if (props.major != 10 || (props.minor != 0 && props.minor != 3)) return 3;
    std::vector<BF16> state(size_t(batch) * D * D), k(size_t(steps) * CHUNK * D), q(k.size());
    for (size_t i = 0; i < state.size(); ++i)
        state[i] = BF16(float((i % (D * D) * 13 + i / (D * D) * 7) % 31) / 32.f - 15.f / 32.f);
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = BF16(float((i * 7 + i / (CHUNK * D) * 3) % 17) / 32.f - 8.f / 32.f);
        q[i] = BF16(float((i * 11 + i / (CHUNK * D) * 5) % 19) / 32.f - 9.f / 32.f);
    }
    BF16 *ds, *dk, *dq, *dr;
    float *dyk, *dyq;
    const size_t count = size_t(batch) * steps * CHUNK * D;
    CHECK_CUDA(cudaMalloc(&ds, state.size() * sizeof(BF16)));
    CHECK_CUDA(cudaMalloc(&dr, state.size() * sizeof(BF16)));
    CHECK_CUDA(cudaMalloc(&dk, k.size() * sizeof(BF16)));
    CHECK_CUDA(cudaMalloc(&dq, q.size() * sizeof(BF16)));
    CHECK_CUDA(cudaMalloc(&dyk, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dyq, count * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(ds, state.data(), state.size() * sizeof(BF16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dk, k.data(), k.size() * sizeof(BF16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dq, q.data(), q.size() * sizeof(BF16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dr, 0xff, state.size() * sizeof(BF16)));
    CHECK_CUDA(cudaMemset(dyk, 0xff, count * sizeof(float)));
    CHECK_CUDA(cudaMemset(dyq, 0xff, count * sizeof(float)));
    auto launch = [&]() { phase1_register_state_kernel<<<batch, 128>>>(ds, dk, dq, dyk, dyq, dr, steps); };
    for (int i = 0; i < warmup; ++i) launch();
    CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaEventSynchronize(stop));
    float ms; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    std::vector<BF16> restored(state.size());
    std::vector<float> yk(count), yq(count);
    CHECK_CUDA(cudaMemcpy(restored.data(), dr, state.size() * sizeof(BF16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(yk.data(), dyk, count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(yq.data(), dyq, count * sizeof(float), cudaMemcpyDeviceToHost));
    bool correct = std::memcmp(restored.data(), state.data(), state.size() * sizeof(BF16)) == 0;
    for (size_t i = 0; i < count; ++i) {
        const size_t block = i / (steps * CHUNK * D);
        const size_t step = i / (CHUNK * D) % steps, row = i / D % CHUNK, col = i % D;
        double rk = 0, rq = 0;
        for (int d = 0; d < D; ++d) {
            const double s = float(state[(block * D + col) * D + d]);
            rk += double(float(k[(step * CHUNK + row) * D + d])) * s;
            rq += double(float(q[(step * CHUNK + row) * D + d])) * s;
        }
        // Dyadic fixtures make the products/sums exactly representable in FP32.
        correct = correct && yk[i] == float(rk) && yq[i] == float(rq);
    }
    cudaFuncAttributes attr{}; CHECK_CUDA(cudaFuncGetAttributes(&attr, phase1_register_state_kernel));
    std::printf("{\"implementation\":\"phase1_register_state\",\"correct\":%s,\"steps\":%d,\"batch\":%d,"
                "\"launch_us\":%.6f,\"registers_per_thread\":%d,\"static_smem_bytes\":%zu,\"local_bytes_per_thread\":%zu}\n",
                correct ? "true" : "false", steps, batch, ms * 1000.0 / iters, attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes);
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(ds)); CHECK_CUDA(cudaFree(dr)); CHECK_CUDA(cudaFree(dk)); CHECK_CUDA(cudaFree(dq));
    CHECK_CUDA(cudaFree(dyk)); CHECK_CUDA(cudaFree(dyq));
    return correct ? 0 : 4;
}
