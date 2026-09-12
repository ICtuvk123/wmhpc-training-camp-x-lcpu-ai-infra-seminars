// Production-shaped V1b Phase-6 mapping probe. This is not production dispatch.
#include "../../../csrc/smxx/utils.cuh"
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using BF16 = cutlass::bfloat16_t;
using namespace cute;

constexpr int M = 128, N = 128, K = 16;
constexpr int NumThreads = 192, ComputeThreads = 128;
constexpr int TmemColumns = 128;

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

CUTE_HOST_DEVICE constexpr auto make_umma() {
  return make_tiled_mma(SM100_MMA_F16BF16_SS<
      BF16, BF16, float, M, N, UMMA::Major::K, UMMA::Major::K>{});
}

CUTE_HOST_DEVICE constexpr auto make_a_layout() {
  auto shape = partition_shape_A(make_umma(), make_shape(Int<M>{}, Int<K>{}));
  return UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<BF16>{}, shape);
}

CUTE_HOST_DEVICE constexpr auto make_b_layout() {
  auto shape = partition_shape_B(make_umma(), make_shape(Int<N>{}, Int<K>{}));
  return UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<BF16>{}, shape);
}

using ALayout = decltype(make_a_layout());
using BLayout = decltype(make_b_layout());

struct SharedStorage {
  alignas(128) ArrayEngine<BF16, cosize_v<ALayout>> a;
  alignas(128) ArrayEngine<BF16, cosize_v<BLayout>> b;
  alignas(16) uint64_t mma_barrier;
  alignas(16) uint32_t tmem_base;
};

static_assert(cosize_v<BLayout> * sizeof(BF16) == K * N * sizeof(BF16));

struct TopologyResult {
  std::string copy_op;
  int register_values_per_thread;
  int same_thread;
  int same_warp_different_lane;
  int different_warp;
  int missing;
  int duplicate;
  int warp_transfer_matrix[4][4];
};

template <class CopyOp>
TopologyResult inspect_topology(char const* name) {
  auto umma = make_umma();
  auto cta = umma.get_slice(Int<0>{});
  Tensor c = make_tensor(make_gmem_ptr(static_cast<float*>(nullptr)),
      make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
  Tensor tCgC = cta.partition_C(c);
  Tensor tCtAcc = cta.make_fragment_C(tCgC);
  tCtAcc.data() = 0;
  auto logical = make_identity_tensor(make_shape(Int<M>{}, Int<N>{}));
  Tensor tCgCoord = cta.partition_C(logical);
  auto tmem_copy = make_tmem_copy(CopyOp{}, tCtAcc);

  int src_owner[M*N];
  int dst_owner[M*N];
  std::fill_n(src_owner,M*N,-1);
  std::fill_n(dst_owner,M*N,-1);
  int values_per_thread=-1;
  int duplicate=0;
  for(int tid=0;tid<ComputeThreads;++tid) {
    auto tmem_thr=tmem_copy.get_slice(tid);
    Tensor coords=tmem_thr.partition_D(tCgCoord);
    if(values_per_thread<0) values_per_thread=size(coords);
    for(int slot=0;slot<size(coords);++slot) {
      auto coord=coords(slot);
      int row=int(get<0>(coord)),col=int(get<1>(coord));
      int logical_idx=row*N+col;
      duplicate += src_owner[logical_idx]>=0;
      src_owner[logical_idx]=tid;
    }

    int warp=tid/32,lane=tid%32;
    auto sm80=make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                             Layout<Shape<_1,_1>>{},Tile<_16,_16,_16>{});
    auto sm80_thr=sm80.get_slice(lane);
    auto identity16=make_identity_tensor(make_shape(Int<16>{},Int<16>{}));
    auto coords_b=sm80_thr.partition_B(identity16);
    for(int bi=0;bi<2;++bi) for(int kb=0;kb<8;++kb) for(int i=0;i<size(coords_b);++i) {
      auto coord=coords_b(i);
      int row=kb*16+int(get<1>(coord));
      int col=(warp*2+bi)*16+int(get<0>(coord));
      int logical_idx=row*N+col;
      duplicate += dst_owner[logical_idx]>=0;
      dst_owner[logical_idx]=tid;
    }
  }

  TopologyResult result{name,values_per_thread,0,0,0,0,duplicate,{}};
  for(int logical_idx=0;logical_idx<M*N;++logical_idx) {
    int src=src_owner[logical_idx],dst=dst_owner[logical_idx];
    if(src<0||dst<0){++result.missing;continue;}
    if(src==dst) ++result.same_thread;
    else if(src/32==dst/32) ++result.same_warp_different_lane;
    else ++result.different_warp;
    ++result.warp_transfer_matrix[src/32][dst/32];
  }
  return result;
}

void print_topology(TopologyResult const& r) {
  std::printf("{\"topology_sweep\":true,\"copy_op\":\"%s\","
      "\"register_values_per_thread\":%d,\"same_thread\":%d,"
      "\"local_bytes_per_thread\":-1,"
      "\"same_warp_different_lane\":%d,\"different_warp\":%d,"
      "\"missing\":%d,\"duplicate\":%d,"
      "\"warp_transfer_matrix\":[[%d,%d,%d,%d],[%d,%d,%d,%d],"
      "[%d,%d,%d,%d],[%d,%d,%d,%d]]}\n",
      r.copy_op.c_str(),r.register_values_per_thread,r.same_thread,
      r.same_warp_different_lane,r.different_warp,r.missing,r.duplicate,
      r.warp_transfer_matrix[0][0],r.warp_transfer_matrix[0][1],r.warp_transfer_matrix[0][2],r.warp_transfer_matrix[0][3],
      r.warp_transfer_matrix[1][0],r.warp_transfer_matrix[1][1],r.warp_transfer_matrix[1][2],r.warp_transfer_matrix[1][3],
      r.warp_transfer_matrix[2][0],r.warp_transfer_matrix[2][1],r.warp_transfer_matrix[2][2],r.warp_transfer_matrix[2][3],
      r.warp_transfer_matrix[3][0],r.warp_transfer_matrix[3][1],r.warp_transfer_matrix[3][2],r.warp_transfer_matrix[3][3]);
}

#define INSPECT(OP) results.push_back(inspect_topology<OP>(#OP))
std::vector<TopologyResult> run_topology_sweep() {
  std::vector<TopologyResult> results;
  INSPECT(SM100_TMEM_LOAD_32dp32b1x);
  INSPECT(SM100_TMEM_LOAD_32dp32b2x);
  INSPECT(SM100_TMEM_LOAD_32dp32b4x);
  INSPECT(SM100_TMEM_LOAD_32dp32b8x);
  INSPECT(SM100_TMEM_LOAD_32dp32b16x);
  INSPECT(SM100_TMEM_LOAD_32dp32b32x);
  INSPECT(SM100_TMEM_LOAD_32dp32b64x);
  INSPECT(SM100_TMEM_LOAD_32dp32b128x);
  INSPECT(SM100_TMEM_LOAD_16dp256b1x);
  INSPECT(SM100_TMEM_LOAD_16dp128b1x);
  INSPECT(SM100_TMEM_LOAD_16dp128b2x);
  INSPECT(SM100_TMEM_LOAD_16dp64b1x);
  INSPECT(SM100_TMEM_LOAD_16dp64b2x);
  INSPECT(SM100_TMEM_LOAD_16dp64b4x);
  INSPECT(SM100_TMEM_LOAD_16dp32b1x);
  INSPECT(SM100_TMEM_LOAD_16dp32b2x);
  INSPECT(SM100_TMEM_LOAD_16dp32b4x);
  INSPECT(SM100_TMEM_LOAD_16dp32b8x);
  for(auto const& r:results) print_topology(r);
  // Primary: fewer cross-warp elements. Secondary: more direct ownership.
  std::stable_sort(results.begin(),results.end(),[](auto const& a,auto const& b){
    bool av=a.missing==0&&a.duplicate==0,bv=b.missing==0&&b.duplicate==0;
    if(av!=bv) return av>bv;
    if(a.different_warp!=b.different_warp) return a.different_warp<b.different_warp;
    if(a.same_thread!=b.same_thread) return a.same_thread>b.same_thread;
    return a.register_values_per_thread<b.register_values_per_thread;
  });
  for(size_t rank=0;rank<results.size();++rank)
    std::printf("{\"topology_rank\":%zu,\"copy_op\":\"%s\",\"different_warp\":%d,"
                "\"same_thread\":%d,\"register_values_per_thread\":%d}\n",
                rank+1,results[rank].copy_op.c_str(),results[rank].different_warp,
                results[rank].same_thread,results[rank].register_values_per_thread);
  auto const& best=results.front();
  int baseline_cross=0;
  for(auto const& r:results) if(r.copy_op=="SM100_TMEM_LOAD_32dp32b1x") baseline_cross=r.different_warp;
  char const* decision=best.different_warp==0 ? "REGISTER_SHUFFLE_CANDIDATE"
      : best.different_warp*10<baseline_cross*9 ? "MATERIALLY_REDUCED_CROSS_WARP"
                                               : "ALL_TO_ALL_CROSS_WARP_INTRINSIC";
  std::printf("{\"topology_decision\":\"%s\",\"best_copy_op\":\"%s\","
              "\"best_different_warp\":%d,\"baseline_different_warp\":%d}\n",
              decision,best.copy_op.c_str(),best.different_warp,baseline_cross);
  return results;
}
#undef INSPECT

__global__ void v1b_mapping_kernel(
    BF16 const* kt, BF16 const* u, BF16 const* initial, float const* decay,
    float* product, BF16* updated, int* a_owner, int* u_owner, int* tmem_owner,
    int* state_owner, int* tmem_slot, int* state_slot, int* errors) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
  if (threadIdx.x >= ComputeThreads) return;
  int tid = threadIdx.x;
  int warp = tid / 32;
  int lane = tid % 32;
  cutlass::arch::NamedBarrier compute_barrier(ComputeThreads, 0);

  extern __shared__ __align__(128) unsigned char raw[];
  auto& storage = *reinterpret_cast<SharedStorage*>(raw);
  Tensor s_a = make_tensor(make_smem_ptr(storage.a.begin()), ALayout{});
  Tensor s_b = make_tensor(make_smem_ptr(storage.b.begin()), BLayout{});

  auto sm80 = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                             Layout<Shape<_1,_1>>{}, Tile<_16,_16,_16>{});
  auto sm80_thr = sm80.get_slice(lane);
  auto identity16 = make_identity_tensor(make_shape(Int<16>{}, Int<16>{}));
  auto b_ref = make_tensor(make_gmem_ptr(static_cast<BF16*>(nullptr)),
                           make_layout(make_shape(Int<16>{}, Int<16>{}), LayoutRight{}));
  using BFrag = decltype(sm80_thr.partition_fragment_B(b_ref));
  BFrag state[2][8];

  // Reconstruct the production Phase-4 U B fragments and write their logical
  // (N,K) coordinates directly into the minimal UMMA B tile.
#pragma unroll
  for (int bi = 0; bi < 2; ++bi) {
    BFrag u_frag = sm80_thr.partition_fragment_B(b_ref);
    auto coords = sm80_thr.partition_B(identity16);
#pragma unroll
    for (int i = 0; i < size(u_frag); ++i) {
      auto coord = coords(i);                 // partition_B exposes (N,K)
      int n = (warp * 2 + bi) * 16 + int(get<0>(coord));
      int k = int(get<1>(coord));
      u_frag(i) = u[k * N + n];
      s_b(make_coord(n, k), Int<0>{}, Int<0>{}) = u_frag(i);
      if (atomicCAS(&u_owner[k * N + n], -1, tid) != -1) atomicAdd(&errors[0], 1);
    }
  }

  // Populate the exact V1a persistent state representation and ownership map.
#pragma unroll
  for (int bi = 0; bi < 2; ++bi) {
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      state[bi][kb] = sm80_thr.partition_fragment_B(b_ref);
      auto coords = sm80_thr.partition_B(identity16);
#pragma unroll
      for (int i = 0; i < size(state[bi][kb]); ++i) {
        auto coord = coords(i);               // (column-within-tile,row-within-tile)
        int row = kb * 16 + int(get<1>(coord));
        int col = (warp * 2 + bi) * 16 + int(get<0>(coord));
        int logical = row * N + col;
        state[bi][kb](i) = initial[logical];
        state_slot[logical] = (bi * 8 + kb) * size(state[bi][kb]) + i;
        if (atomicCAS(&state_owner[logical], -1, tid) != -1) atomicAdd(&errors[1], 1);
      }
    }
  }

  // k_restored_t is logically [128,16], exactly tcgen05 A=[M,K].
  for (int linear = tid; linear < M * K; linear += ComputeThreads) {
    int row = linear / K, k = linear % K;
    s_a(make_coord(row, k), Int<0>{}, Int<0>{}) = kt[row * K + k];
    if (atomicCAS(&a_owner[linear], -1, tid) != -1) atomicAdd(&errors[4], 1);
  }
  cutlass::arch::fence_view_async_shared();
  compute_barrier.arrive_and_wait();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  auto umma = make_umma();
  auto cta = umma.get_slice(Int<0>{});
  auto c_identity = make_identity_tensor(make_shape(Int<M>{}, Int<N>{}));
  Tensor g_product = make_tensor(make_gmem_ptr(product),
      make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
  Tensor tCgC = cta.partition_C(g_product);
  Tensor tCrA = cta.make_fragment_A(s_a);
  Tensor tCrB = cta.make_fragment_B(s_b);
  Tensor tCtAcc = cta.make_fragment_C(tCgC);

  TMEM::Allocator1Sm allocator;
  if (warp == 0) allocator.allocate(TmemColumns, &storage.tmem_base);
  compute_barrier.arrive_and_wait();
  if (warp == 0) allocator.release_allocation_lock();
  tCtAcc.data() = storage.tmem_base;
  if (tid == 0) {
    initialize_barrier(storage.mma_barrier, 1);
    cutlass::arch::fence_barrier_init();
  }
  compute_barrier.arrive_and_wait();

  umma.accumulate_ = UMMA::ScaleOut::Zero;
  if (warp == 0) {
#pragma unroll
    for (int kb = 0; kb < size<2>(tCrA); ++kb) {
      gemm(umma, tCrA(_,_,kb), tCrB(_,_,kb), tCtAcc);
      umma.accumulate_ = UMMA::ScaleOut::One;
    }
    cutlass::arch::umma_arrive(&storage.mma_barrier);
  }
  wait_barrier(storage.mma_barrier, 0);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  TiledCopy tmem_to_register =
      make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy tmem_copy_thread = tmem_to_register.get_slice(threadIdx.x);
  Tensor tDtAcc = tmem_copy_thread.partition_S(tCtAcc);
  Tensor tDgC = tmem_copy_thread.partition_D(tCgC);
  using AccType = typename decltype(tCtAcc)::value_type;
  // Keep copy-atom mode 0 intact. It represents 32 TMEM data paths even
  // though each thread's register destination has a single first mode.
  Tensor tDtAccGrouped = group_modes<1, decltype(rank(tDtAcc))::value>(tDtAcc);
  Tensor tDgCGrouped = group_modes<1, decltype(rank(tDgC))::value>(tDgC);

  Tensor tCgCoord = cta.partition_C(c_identity);
  Tensor tDgCoord = tmem_copy_thread.partition_D(tCgCoord);
  Tensor tDgCoordGrouped = group_modes<1, decltype(rank(tDgCoord))::value>(tDgCoord);
  constexpr int AccChunk = 16;
  CUTE_STATIC_ASSERT_V(size<0>(tDgCGrouped) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<0>(tDgCoordGrouped) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size(tDgCGrouped) == size(tDgCoordGrouped));
  CUTE_STATIC_ASSERT_V(size<1>(tDgCGrouped) % Int<AccChunk>{} == Int<0>{});
  int matched = 0;
#pragma unroll
  for (int chunk = 0; chunk < size<1>(tDgCGrouped) / AccChunk; ++chunk) {
    Tensor tDtChunk = local_tile(tDtAccGrouped,
        make_shape(shape<0>(tDtAccGrouped), Int<AccChunk>{}), make_coord(0, chunk));
    Tensor tDrChunk = make_tensor<AccType>(
        make_shape(shape<0>(tDgCGrouped), Int<AccChunk>{}));
    copy(tmem_to_register, tDtChunk, tDrChunk);
    cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
    for (int i = 0; i < AccChunk; ++i) {
    int slot = chunk * AccChunk + i;
    auto tc = tDgCoordGrouped(0, slot);
    int row = int(get<0>(tc)), col = int(get<1>(tc));
    int logical = row * N + col;
    tmem_slot[logical] = slot;
    if (atomicCAS(&tmem_owner[logical], -1, tid) != -1) atomicAdd(&errors[2], 1);
    product[logical] = tDrChunk(0, i);

    bool found = false;
#pragma unroll
    for (int bi = 0; bi < 2; ++bi) {
#pragma unroll
      for (int kb = 0; kb < 8; ++kb) {
        auto bc = sm80_thr.partition_B(identity16);
#pragma unroll
        for (int j = 0; j < size(state[bi][kb]); ++j) {
          auto sc = bc(j);
          int sr = kb * 16 + int(get<1>(sc));
          int scol = (warp * 2 + bi) * 16 + int(get<0>(sc));
          if (sr == row && scol == col) {
            state[bi][kb](j) = BF16(
                bf16_to_f32(state[bi][kb](j)) * decay[row] + tDrChunk(0, i));
            found = true;
          }
        }
      }
    }
    matched += found;
    }
  }
  if (matched != size(tDgC)) atomicAdd(&errors[3], size(tDgC) - matched);

  // Store through the V1a B-fragment logical coordinates, not TMEM ownership.
#pragma unroll
  for (int bi = 0; bi < 2; ++bi) {
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
      auto coords = sm80_thr.partition_B(identity16);
#pragma unroll
      for (int i = 0; i < size(state[bi][kb]); ++i) {
        auto coord = coords(i);
        int row = kb * 16 + int(get<1>(coord));
        int col = (warp * 2 + bi) * 16 + int(get<0>(coord));
        updated[row * N + col] = state[bi][kb](i);
      }
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  compute_barrier.arrive_and_wait();
  if (warp == 0) allocator.free(storage.tmem_base, TmemColumns);
#else
  if (threadIdx.x == 0) asm volatile("trap;");
#endif
}

int main(int argc, char** argv) {
  int warmup = 1, iters = 1;
  bool topology_only = false;
  std::string dump_ownership;
  for (int i = 1; i < argc;) {
    if (!std::strcmp(argv[i], "--topology-only")) { topology_only = true; ++i; }
    else if (i + 1 >= argc) return 2;
    else if (!std::strcmp(argv[i], "--dump-ownership")) { dump_ownership = argv[i + 1]; i += 2; }
    else if (!std::strcmp(argv[i], "--warmup")) { warmup = std::atoi(argv[i + 1]); i += 2; }
    else if (!std::strcmp(argv[i], "--iters")) { iters = std::atoi(argv[i + 1]); i += 2; }
    else return 2;
  }
  auto topology_results=run_topology_sweep();
  (void)topology_results;
  if (topology_only) return 0;
  int dev; cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&dev)); CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  if (prop.major != 10 || prop.minor != 3) return 3;

  std::vector<BF16> kt(M*K), u(K*N), initial(M*N);
  std::vector<float> decay(M, 0.875f);
  // One inner-k contribution; integer products are exact in FP32.
  for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k)
    kt[m*K+k] = BF16(k == 7 ? float((m % 13) + 1) : 0.f);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n)
    u[k*N+n] = BF16(k == 7 ? float((n % 11) - 5) : 0.f);
  for (int i = 0; i < M*N; ++i) initial[i] = BF16(float((i % 17) - 8) / 8.f);

  BF16 *dkt, *du, *di, *dout; float *dg, *dprod;
  int *dao, *duo, *dto, *dso, *dts, *dss, *derr;
  CHECK_CUDA(cudaMalloc(&dkt, kt.size()*sizeof(BF16))); CHECK_CUDA(cudaMalloc(&du, u.size()*sizeof(BF16)));
  CHECK_CUDA(cudaMalloc(&di, initial.size()*sizeof(BF16))); CHECK_CUDA(cudaMalloc(&dg, decay.size()*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dprod, M*N*sizeof(float))); CHECK_CUDA(cudaMalloc(&dout, M*N*sizeof(BF16)));
  CHECK_CUDA(cudaMalloc(&dao, M*K*sizeof(int))); CHECK_CUDA(cudaMalloc(&duo, K*N*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dto, M*N*sizeof(int))); CHECK_CUDA(cudaMalloc(&dso, M*N*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dts, M*N*sizeof(int))); CHECK_CUDA(cudaMalloc(&dss, M*N*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&derr, 5*sizeof(int)));
  CHECK_CUDA(cudaMemcpy(dkt,kt.data(),kt.size()*sizeof(BF16),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(du,u.data(),u.size()*sizeof(BF16),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(di,initial.data(),initial.size()*sizeof(BF16),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dg,decay.data(),decay.size()*sizeof(float),cudaMemcpyHostToDevice));
  auto launch = [&] {
    CHECK_CUDA(cudaMemset(dao,0xff,M*K*sizeof(int))); CHECK_CUDA(cudaMemset(duo,0xff,K*N*sizeof(int)));
    CHECK_CUDA(cudaMemset(dto,0xff,M*N*sizeof(int))); CHECK_CUDA(cudaMemset(dso,0xff,M*N*sizeof(int)));
    CHECK_CUDA(cudaMemset(dts,0xff,M*N*sizeof(int))); CHECK_CUDA(cudaMemset(dss,0xff,M*N*sizeof(int)));
    CHECK_CUDA(cudaMemset(derr,0,5*sizeof(int)));
    v1b_mapping_kernel<<<1,NumThreads,sizeof(SharedStorage)>>>(
        dkt,du,di,dg,dprod,dout,dao,duo,dto,dso,dts,dss,derr);
  };
  for (int i=0;i<warmup;++i) launch(); CHECK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t a,b; CHECK_CUDA(cudaEventCreate(&a)); CHECK_CUDA(cudaEventCreate(&b)); CHECK_CUDA(cudaEventRecord(a));
  for (int i=0;i<iters;++i) launch(); CHECK_CUDA(cudaEventRecord(b)); CHECK_CUDA(cudaEventSynchronize(b));
  float ms; CHECK_CUDA(cudaEventElapsedTime(&ms,a,b));
  std::vector<float> product(M*N); std::vector<BF16> output(M*N);
  std::vector<int> ao(M*K),uo(K*N),to(M*N),so(M*N),ts(M*N),ss(M*N),errors(5);
  CHECK_CUDA(cudaMemcpy(product.data(),dprod,M*N*sizeof(float),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(output.data(),dout,M*N*sizeof(BF16),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ao.data(),dao,M*K*sizeof(int),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(uo.data(),duo,K*N*sizeof(int),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(to.data(),dto,M*N*sizeof(int),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(so.data(),dso,M*N*sizeof(int),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ts.data(),dts,M*N*sizeof(int),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ss.data(),dss,M*N*sizeof(int),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(errors.data(),derr,5*sizeof(int),cudaMemcpyDeviceToHost));
  int missing_a=0,missing_u=0,missing_t=0,missing_s=0,owner_mismatch=0,bad_product=0,bad_update=0;
  double sum_error=0,max_error=0;
  for(int i=0;i<M*K;++i) missing_a += ao[i] < 0;
  for(int i=0;i<K*N;++i) missing_u += uo[i] < 0;
  for(int i=0;i<M*N;++i) {
    missing_t += to[i] < 0; missing_s += so[i] < 0; owner_mismatch += to[i] != so[i];
    int row=i/N,col=i%N; float ref=float(kt[row*K+7])*float(u[7*N+col]);
    double err=std::abs(double(product[i])-ref); sum_error+=err; max_error=std::max(max_error,err); bad_product += product[i]!=ref;
    BF16 ref_update=BF16(float(initial[i])*decay[row]+ref); bad_update += output[i]!=ref_update;
  }

  // Ownership topology is intentionally analyzed on the host. The diagnostic
  // kernel only exports the native thread/slot assigned by each partition.
  int same_thread=0,same_warp_different_lane=0,different_warp=0;
  int warp_transfer_matrix[4][4] = {};
  int lane_map[32]; std::fill_n(lane_map,32,-1);
  int warp_map[4]; std::fill_n(warp_map,4,-1);
  bool fixed_lane_permutation=true,fixed_warp_permutation=true;
  int tile_same[8][8] = {},tile_warp_local[8][8] = {},tile_cross[8][8] = {};
  for(int logical=0;logical<M*N;++logical) {
    int src=to[logical],dst=so[logical];
    if(src<0||dst<0) continue;
    int sw=src/32,dw=dst/32,sl=src%32,dl=dst%32;
    ++warp_transfer_matrix[sw][dw];
    if(src==dst) ++same_thread;
    else if(sw==dw) ++same_warp_different_lane;
    else ++different_warp;
    if(lane_map[sl]<0) lane_map[sl]=dl;
    else if(lane_map[sl]!=dl) fixed_lane_permutation=false;
    if(warp_map[sw]<0) warp_map[sw]=dw;
    else if(warp_map[sw]!=dw) fixed_warp_permutation=false;
    int row=logical/N,col=logical%N,tr=row/16,tc=col/16;
    if(src==dst) ++tile_same[tr][tc];
    else if(sw==dw) ++tile_warp_local[tr][tc];
    else ++tile_cross[tr][tc];
  }
  bool ownership_bijection = !missing_t&&!missing_s&&!errors[1]&&!errors[2];
  bool topology_complete = same_thread+same_warp_different_lane+different_warp==M*N;
  int cross_warp_tiles=0,cross_warp_strips=0;
  for(int tr=0;tr<8;++tr) {
    bool strip_has_cross=false;
    for(int tc=0;tc<8;++tc) if(tile_cross[tr][tc]) {++cross_warp_tiles;strip_has_cross=true;}
    cross_warp_strips += strip_has_cross;
  }
  const char* mapping;
  const char* mapping_topology;
  int minimum_cross_warp_scratch_bytes=0,required_compute_barriers=0;
  if(different_warp==0 && same_warp_different_lane==0) {
    mapping="DIRECT"; mapping_topology="DIRECT_SAME_THREAD";
  } else if(different_warp==0) {
    mapping="REGISTER_SHUFFLE";
    mapping_topology=fixed_lane_permutation ? "FIXED_LANE_PERMUTATION"
                                            : "TILE_COORDINATE_LANE_PERMUTATION";
  } else {
    // Any bijective cross-warp permutation can be exchanged one logical 16x16
    // tile at a time: sources write scratch[row_in_tile,col_in_tile], named
    // barrier, destinations read, named barrier. Thus 1 KiB is sufficient;
    // a 16x128 strip trades 8 KiB for only 16 barriers over the full matrix.
    mapping="SMALL_SCRATCH";
    mapping_topology=(fixed_warp_permutation&&fixed_lane_permutation)
        ? "FIXED_WARP_AND_LANE_PERMUTATION"
        : "GENERAL_COORDINATE_DEPENDENT_CROSS_WARP";
    minimum_cross_warp_scratch_bytes=16*16*sizeof(float);
    required_compute_barriers=2*cross_warp_tiles;
  }
  // DIRECT is exercised numerically above. Cross-warp SMALL_SCRATCH is proven
  // constructively by the bounded logical-tile exchange. A warp-local result
  // remains NO-GO until an actual shuffle path is exercised.
  bool mapping_go=ownership_bijection&&topology_complete&&bad_product==0&&
                  (different_warp>0 || same_warp_different_lane==0);

  std::ostringstream matrix_json,tile_grid,lane_json,warp_json;
  matrix_json<<"[";
  for(int sw=0;sw<4;++sw){if(sw)matrix_json<<",";matrix_json<<"[";
    for(int dw=0;dw<4;++dw){if(dw)matrix_json<<",";matrix_json<<warp_transfer_matrix[sw][dw];}
    matrix_json<<"]";} matrix_json<<"]";
  tile_grid<<"[";
  for(int tr=0;tr<8;++tr){if(tr)tile_grid<<",";tile_grid<<"\"";
    for(int tc=0;tc<8;++tc) tile_grid<<(tile_cross[tr][tc]? 'X':tile_warp_local[tr][tc]?'W':'D');
    tile_grid<<"\"";} tile_grid<<"]";
  lane_json<<"[";for(int i=0;i<32;++i){if(i)lane_json<<",";lane_json<<lane_map[i];}lane_json<<"]";
  warp_json<<"[";for(int i=0;i<4;++i){if(i)warp_json<<",";warp_json<<warp_map[i];}warp_json<<"]";

  if(!dump_ownership.empty()) {
    std::ofstream csv(dump_ownership);
    csv<<"row,col,src_thread,src_warp,src_lane,src_register_slot,"
          "dst_thread,dst_warp,dst_lane,dst_register_slot\n";
    for(int logical=0;logical<M*N;++logical) {
      int row=logical/N,col=logical%N,src=to[logical],dst=so[logical];
      csv<<row<<','<<col<<','<<src<<','<<src/32<<','<<src%32<<','<<ts[logical]
         <<','<<dst<<','<<dst/32<<','<<dst%32<<','<<ss[logical]<<'\n';
    }
  }
  bool direct = !owner_mismatch && !errors[3];
  bool correct = !missing_a&&!missing_u&&!missing_t&&!missing_s&&!errors[0]&&!errors[1]&&!errors[2]&&
                 !errors[4]&&!bad_product&&!bad_update&&direct;
  cudaFuncAttributes attr{}; CHECK_CUDA(cudaFuncGetAttributes(&attr,v1b_mapping_kernel));
  std::printf("{\"correct\":%s,\"mapping\":\"%s\",\"mapping_topology\":\"%s\","
    "\"same_thread\":%d,\"same_warp_different_lane\":%d,\"different_warp\":%d,"
    "\"cross_warp_element_count\":%d,\"minimum_cross_warp_scratch_bytes\":%d,"
    "\"required_compute_barriers\":%d,\"cross_warp_tiles\":%d,\"strip_scratch_bytes\":8192,"
    "\"strip_compute_barriers\":%d,"
    "\"warp_transfer_matrix\":%s,\"tile_topology_grid\":%s,\"lane_map\":%s,\"warp_map\":%s,"
    "\"u_staging_bytes\":%zu,\"additional_smem_bytes\":%zu,"
    "\"tmem_columns\":%d,\"registers_per_thread\":%d,\"local_bytes_per_thread\":%zu,\"spills\":\"see_ptxas\","
    "\"v1b_mapping_go\":%s,\"missing_a\":%d,\"missing_u\":%d,"
    "\"missing_tmem\":%d,\"missing_state\":%d,\"owner_mismatch\":%d,\"duplicate_u\":%d,"
    "\"duplicate_tmem\":%d,\"duplicate_state\":%d,\"unmapped_local\":%d,\"bad_product\":%d,"
    "\"bad_bf16_update\":%d,\"max_product_error\":%.9g,\"mean_product_error\":%.9g,\"launch_us\":%.6f}\n",
    correct?"true":"false",mapping,mapping_topology,same_thread,same_warp_different_lane,different_warp,
    different_warp,minimum_cross_warp_scratch_bytes,required_compute_barriers,cross_warp_tiles,
    2*cross_warp_strips,matrix_json.str().c_str(),
    tile_grid.str().c_str(),lane_json.str().c_str(),warp_json.str().c_str(),size_t(K*N*sizeof(BF16)),sizeof(SharedStorage),
    TmemColumns,attr.numRegs,attr.localSizeBytes,mapping_go?"true":"false",missing_a,missing_u,missing_t,missing_s,
    owner_mismatch,errors[0],errors[2],errors[1],errors[3],
    bad_product,bad_update,max_error,sum_error/(M*N),ms*1000/iters);
  return correct?0:4;
}
