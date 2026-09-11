// Host-only layout regression. Compiles the real P3 source but launches no GPU
// work. This checks chunk addressing and decay mapping, not CUDA execution.
#define main p3_benchmark_main
#include "tcgen05_p3_recurrence.cu"
#undef main
#include <cassert>

int main() {
  using namespace cute;
  auto mma = make_tcgen05_mma();
  auto tiler = make_tcgen05_tiler();
  auto cta = mma.get_slice(Int<0>{});
  std::vector<float> output(kM * kN), decay(kM);
  std::vector<int> visits(kM * kN, 0);
  for (int i = 0; i < kM * kN; ++i) output[i] = float(i);
  for (int m = 0; m < kM; ++m) decay[m] = float(m);
  Tensor m_c = make_tensor(make_gmem_ptr(output.data()),
      make_layout(make_shape(Int<kM>{}, Int<kN>{}), make_stride(Int<kN>{}, Int<1>{})));
  Tensor m_g = make_tensor(make_gmem_ptr(decay.data()),
      make_layout(make_shape(Int<kM>{}, Int<kN>{}), make_stride(Int<1>{}, Int<0>{})));
  auto coord = make_coord(0, 0, _);
  Tensor g_c = local_tile(m_c, tiler, coord, Step<_1, _1, X>{});
  Tensor g_g = local_tile(m_g, tiler, coord, Step<_1, _1, X>{});
  Tensor tCgC = cta.partition_C(g_c);
  Tensor tCgG = cta.partition_C(g_g);
  Tensor tCtAcc = cta.make_fragment_C(tCgC);
  tCtAcc.data() = 0;
  auto tmem_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  size_t checked_addresses = 0;
  for (int tid = 0; tid < kThreads; ++tid) {
    auto thr = tmem_copy.get_slice(tid);
    Tensor src = thr.partition_S(tCtAcc);
    Tensor dst = thr.partition_D(tCgC);
    Tensor g = thr.partition_D(tCgG);
    Tensor src_group = group_modes<1, decltype(rank(src))::value>(src);
    Tensor dst_group = group_modes<1, decltype(rank(dst))::value>(dst);
    Tensor g_group = group_modes<1, decltype(rank(g))::value>(g);
    Tensor state = make_tensor<float>(shape(dst));
    Tensor state_group = group_modes<1, decltype(rank(state))::value>(state);
    copy(dst, state);
    for (int chunk = 0; chunk < size<1>(state_group) / 16; ++chunk) {
      Tensor src_chunk = local_tile(src_group,
          make_shape(shape<0>(src_group), Int<16>{}), make_coord(0, chunk));
      for (int i = 0; i < 16; ++i) {
        const int index = chunk * 16 + i;
        const int element = int(dst(index));
        assert(dst_group(0, index) == dst(index));
        assert(state_group(0, index) == dst(index));
        assert(g_group(0, index) == float(element / kN));
        assert(g_group(0, index) == g(index));
        ++visits[element];
        // Source mode 0 is collective (32 data paths), unlike the per-thread
        // destination's one value. Check every address of each TMEM chunk.
        for (int v = 0; v < size<0>(src_group); ++v) {
          const auto chunk_address = raw_pointer_cast(src_chunk.data()) + src_chunk.layout()(v, i);
          const auto original_address = raw_pointer_cast(src.data()) +
              src.layout()(v + size<0>(src_group) * index);
          assert(chunk_address == original_address);
          ++checked_addresses;
        }
      }
    }
  }
  for (int count : visits) assert(count == 1);
  std::printf("PASS: %d threads, %d output elements covered once, %zu TMEM addresses; row decay preserved\n",
              kThreads, kM * kN, checked_addresses);
}
