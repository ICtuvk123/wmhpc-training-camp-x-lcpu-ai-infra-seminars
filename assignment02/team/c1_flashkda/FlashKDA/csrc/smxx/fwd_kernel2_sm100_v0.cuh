#pragma once

// Opt-in production Phase-6 experiment. Canonical state remains BF16 SMEM.
// Included only when FLASH_KDA_ENABLE_SM100_V0 is enabled at build time.
#include <cute/arch/tmem_allocator_sm100.hpp>

namespace flash_kda_sm100_v0 {
using BF16 = cutlass::bfloat16_t;

CUTE_HOST_DEVICE constexpr auto make_mma() {
    return cute::make_tiled_mma(cute::SM100_MMA_F16BF16_SS<
        BF16, BF16, float, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>{});
}

CUTE_HOST_DEVICE constexpr auto operand_layout() {
    using namespace cute;
    auto shape = partition_shape_A(make_mma(), make_shape(Int<128>{}, Int<16>{}));
    return UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<BF16>{}, shape);
}

using OperandLayout = decltype(operand_layout());

template <class VOLayout>
struct Storage {
    // U arrives in the existing SM80 C fragment. First store using the existing
    // STSM/VO layout, then re-layout into UMMA B. V0 prioritizes correctness.
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> u;
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<OperandLayout>> a;
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<OperandLayout>> b;
    alignas(16) uint64_t mma_barrier;
    alignas(16) uint32_t tmem_base;
};

template <class StorageT>
CUTE_DEVICE void initialize(StorageT& storage, int tid,
                            cutlass::arch::NamedBarrier& compute_barrier) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    cute::TMEM::Allocator1Sm allocator;
    if (tid / 32 == 0) allocator.allocate(128, &storage.tmem_base);
    compute_barrier.arrive_and_wait();
    if (tid / 32 == 0) allocator.release_allocation_lock();
    if (tid == 0) {
        cute::initialize_barrier(storage.mma_barrier, 1);
        cutlass::arch::fence_barrier_init();
    }
    compute_barrier.arrive_and_wait();
#else
    asm volatile("trap;");
#endif
}

template <class StorageT>
CUTE_DEVICE void release(StorageT& storage, int tid,
                         cutlass::arch::NamedBarrier& compute_barrier) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    compute_barrier.arrive_and_wait();
    if (tid / 32 == 0) {
        cute::TMEM::Allocator1Sm allocator;
        allocator.free(storage.tmem_base, 128);
    }
#else
    asm volatile("trap;");
#endif
}

template <class StorageT, class KR, class U, class State, class Decay>
CUTE_DEVICE void update(StorageT& storage, KR const& kr_transposed,
                        U const& u_transposed, State& state_transposed,
                        Decay const& g_total, int tid, int step,
                        cutlass::arch::NamedBarrier& compute_barrier) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    using namespace cute;
    auto mma = make_mma();
    auto cta = mma.get_slice(Int<0>{});
    Tensor s_a = make_tensor(make_smem_ptr(storage.a.begin()), OperandLayout{});
    Tensor s_b = make_tensor(make_smem_ptr(storage.b.begin()), OperandLayout{});
    Tensor tCgA = cta.partition_A(kr_transposed);
    Tensor tCgB = cta.partition_B(u_transposed);
    Tensor tCgS = cta.partition_C(state_transposed);
    Tensor tCrA = cta.make_fragment_A(s_a);
    Tensor tCrB = cta.make_fragment_B(s_b);
    Tensor tCtAcc = cta.make_fragment_C(tCgS);
    tCtAcc.data() = storage.tmem_base;

    cooperative_copy<128>(tid, tCgA, s_a);
    cooperative_copy<128>(tid, tCgB, s_b);
    cutlass::arch::fence_view_async_shared();
    compute_barrier.arrive_and_wait();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    mma.accumulate_ = UMMA::ScaleOut::Zero;
    if (tid / 32 == 0) {
#pragma unroll
        for (int k = 0; k < size<2>(tCrA); ++k) {
            gemm(mma, tCrA(_, _, k), tCrB(_, _, k), tCtAcc);
            mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&storage.mma_barrier);
    }
    cute::wait_barrier(storage.mma_barrier, step & 1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    auto copy_tmem = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
    auto thr_copy = copy_tmem.get_slice(tid);
    Tensor tDtAcc = thr_copy.partition_S(tCtAcc);
    Tensor tDsS = thr_copy.partition_D(tCgS);
    Tensor acc = make_tensor<float>(shape(tDsS));
    copy(copy_tmem, tDtAcc, acc);
    cutlass::arch::fence_view_async_tmem_load();

    // Match production: g scales key-feature rows in the transposed state
    // view; round every chunk's result to BF16 even for FP32 state I/O.
    Tensor s_g = make_tensor(g_total.data(),
        make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(Int<1>{}, Int<0>{})));
    Tensor tCgG = cta.partition_C(s_g);
    Tensor tDsG = thr_copy.partition_D(tCgG);
    Tensor result = make_tensor<BF16>(shape(tDsS));
#pragma unroll
    for (int i = 0; i < size(acc); ++i) {
        result(i) = BF16(bf16_to_f32(tDsS(i)) * tDsG(i) + acc(i));
    }
    copy(result, tDsS);
    // Caller immediately executes the existing 128-thread compute barrier.
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
#else
    asm volatile("trap;");
#endif
}
}  // namespace flash_kda_sm100_v0
