// Host-only checks of production/V0 layout contracts and the proposed SM80
// C-to-B register permutation. No CUDA operations are launched.
#define FLASH_KDA_ENABLE_SM100_V0 1
#include "../../csrc/smxx/fwd_kernel2.cuh"
#include <array>
#include <cassert>

int main() {
    using namespace cute;
    using BF16 = cutlass::bfloat16_t;
    using Layouts = K2Layouts<128, 16>;
    using Baseline = SharedStorageK2<Layouts, 3, 2>;
    using V0 = SharedStorageK2SM100V0<Layouts, 3, 2>;
    using V1A = SharedStorageK2V1A<Layouts, 3, 2>;
    alignas(128) ArrayEngine<BF16, cosize_v<Layouts::StateSmemLayout>> state_data;
    alignas(128) ArrayEngine<BF16, cosize_v<Layouts::MMALayout>> kr_data;
    flash_kda_sm100_v0::Storage<Layouts::VOLayout> scratch;
    Tensor state = make_tensor(make_smem_ptr(state_data.begin()), Layouts::StateSmemLayout{});
    Tensor state_t = make_tensor(make_smem_ptr(state_data.begin()), Layouts::TransposedStateSmemLayout{});
    Tensor u = make_tensor(make_smem_ptr(scratch.u.begin()), Layouts::VOLayout{});
    Tensor ut = make_tensor(make_smem_ptr(scratch.u.begin()), Layouts::TransposedVOLayout{});
    Tensor kr = make_tensor(make_smem_ptr(kr_data.begin()), Layouts::MMALayout{});
    Tensor krt = make_tensor(make_smem_ptr(kr_data.begin()), Layouts::TransposedMMALayout{});
    for (int m = 0; m < 128; ++m) for (int n = 0; n < 128; ++n) {
        BF16 value; value.storage = uint16_t(m * 128 + n);
        state_t(m, n) = value;
    }
    for (int t = 0; t < 16; ++t) for (int n = 0; n < 128; ++n) {
        BF16 value; value.storage = uint16_t(t * 128 + n);
        u(t, n) = kr(t, n) = value;
        assert(ut(n, t).storage == value.storage);
        assert(krt(n, t).storage == value.storage);
    }
    for (int m = 0; m < 128; ++m) for (int n = 0; n < 128; ++n)
        assert(state(n, m).storage == state_t(m, n).storage);
    auto mma100 = flash_kda_sm100_v0::make_mma();
    auto cta = mma100.get_slice(Int<0>{});
    Tensor tCgS = cta.partition_C(state_t);
    Tensor acc = cta.make_fragment_C(tCgS);
    acc.data() = 0;
    auto tmem_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, acc);
    std::array<float, 128> decay;
    for (int m = 0; m < 128; ++m) decay[m] = float(m);
    Tensor g = make_tensor(decay.data(), make_layout(make_shape(_128{}, _128{}), make_stride(_1{}, _0{})));
    Tensor tCgG = cta.partition_C(g);
    std::array<int, 128 * 128> visits{};
    for (int tid = 0; tid < 128; ++tid) {
        auto thr = tmem_copy.get_slice(tid);
        Tensor s_part = thr.partition_D(tCgS);
        Tensor g_part = thr.partition_D(tCgG);
        for (int i = 0; i < size(s_part); ++i) {
            const unsigned logical = s_part(i).storage;
            assert(g_part(i) == float(logical / 128));
            ++visits[logical];
        }
    }
    for (int count : visits) assert(count == 1);

    // PTX movmatrix m8n8 transposes each lane's packed two BF16 elements
    // across a warp. Emulate the specified permutation on coordinate tags,
    // then compare with CuTe's independently defined MMA B partition.
    auto mma80 = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                                Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
    Tensor identity = make_identity_tensor(make_shape(_16{}, _16{}));
    int c_values[32][8], b_values[32][8];
    for (int lane = 0; lane < 32; ++lane) {
        auto thr = mma80.get_slice(lane);
        Tensor c = thr.partition_C(identity);
        Tensor b = thr.partition_B(identity);
        for (int i = 0; i < 8; ++i) {
            auto cc = c(i), bc = b(i);
            c_values[lane][i] = int(get<0>(cc)) * 16 + int(get<1>(cc)); // key,value
            // For this TN atom CuTe exposes the B identity coordinate as
            // (N,K), so convert it to the shared logical (K,N) tag.
            b_values[lane][i] = int(get<1>(bc)) * 16 + int(get<0>(bc));
        }
    }
    int b_from_c[32][8], c_roundtrip[32][8];
    for (int lane = 0; lane < 32; ++lane) for (int word = 0; word < 4; ++word)
        for (int half = 0; half < 2; ++half) {
            const int row = lane / 4, col = (lane % 4) * 2 + half;
            const int source_lane = col * 4 + row / 2, source_half = row % 2;
            b_from_c[lane][word * 2 + half] = c_values[source_lane][word * 2 + source_half];
            assert(b_from_c[lane][word * 2 + half] == b_values[lane][word * 2 + half]);
        }
    // MOVM_T is the inverse distribution change as well: B -> C -> B (and
    // C -> B -> C) must preserve every coordinate-coded element bit-for-bit.
    for (int lane = 0; lane < 32; ++lane) for (int word = 0; word < 4; ++word)
        for (int half = 0; half < 2; ++half) {
            const int row = lane / 4, col = (lane % 4) * 2 + half;
            const int source_lane = col * 4 + row / 2, source_half = row % 2;
            c_roundtrip[lane][word * 2 + half] = b_from_c[source_lane][word * 2 + source_half];
            assert(c_roundtrip[lane][word * 2 + half] == c_values[lane][word * 2 + half]);
        }
    // Direct-GMEM V1a load/store must use the same (K,N) tags as baseline
    // C->MOVM_T->B. This catches the former tile-local transpose explicitly.
    std::array<int, 128 * 128> register_visits{};
    for (int warp = 0; warp < 4; ++warp) for (int bi = 0; bi < 2; ++bi)
        for (int kb = 0; kb < 8; ++kb) for (int lane = 0; lane < 32; ++lane)
            for (int i = 0; i < 8; ++i) {
                int key = kb * 16 + b_values[lane][i] / 16;
                int value = (warp * 2 + bi) * 16 + b_values[lane][i] % 16;
                ++register_visits[value * 128 + key];
            }
    for (int count : register_visits) assert(count == 1);
    std::printf("PASS: state/U/K aliases, V0 decay partition, MOVM mapping, V1a full-state ownership\n"
                "baseline_smem_bytes=%zu v0_smem_bytes=%zu v1a_smem_bytes=%zu v0_extra_bytes=%zu v1a_saved_bytes=%zu\n",
                sizeof(Baseline), sizeof(V0), sizeof(V1A), sizeof(V0) - sizeof(Baseline), sizeof(Baseline) - sizeof(V1A));
}
