// Production-shaped V1b Phase-6 mapping probe. This is not production dispatch.
#include "../../../csrc/smxx/utils.cuh"
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

__global__ void v1b_mapping_kernel(
    BF16 const* kt, BF16 const* u, BF16 const* initial, float const* decay,
    float* product, BF16* updated, int* a_owner, int* u_owner, int* tmem_owner,
    int* state_owner, int* errors) {
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
      s_b(n, k) = u_frag(i);
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
        if (atomicCAS(&state_owner[logical], -1, tid) != -1) atomicAdd(&errors[1], 1);
      }
    }
  }

  // k_restored_t is logically [128,16], exactly tcgen05 A=[M,K].
  for (int linear = tid; linear < M * K; linear += ComputeThreads) {
    int row = linear / K, k = linear % K;
    s_a(row, k) = kt[row * K + k];
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

  auto tmem_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto tmem_thr = tmem_copy.get_slice(tid);
  Tensor tDtAcc = tmem_thr.partition_S(tCtAcc);
  Tensor tDgC = tmem_thr.partition_D(tCgC);
  Tensor tCgCoord = cta.partition_C(c_identity);
  Tensor tDgCoord = tmem_thr.partition_D(tCgCoord);
  Tensor tDtGrouped = group_modes<1, decltype(rank(tDtAcc))::value>(tDtAcc);
  Tensor tDgCoordGrouped = group_modes<1, decltype(rank(tDgCoord))::value>(tDgCoord);
  constexpr int AccChunk = 16;
  CUTE_STATIC_ASSERT_V(size<0>(tDtGrouped) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<1>(tDtGrouped) % Int<AccChunk>{} == Int<0>{});
  int matched = 0;
#pragma unroll
  for (int chunk = 0; chunk < size<1>(tDtGrouped) / AccChunk; ++chunk) {
    Tensor tDtChunk = local_tile(tDtGrouped, make_shape(_1{}, Int<AccChunk>{}),
                                 make_coord(0, chunk));
    Tensor acc = make_tensor<float>(make_shape(_1{}, Int<AccChunk>{}));
    copy(tmem_copy, tDtChunk, acc);
    cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
    for (int i = 0; i < AccChunk; ++i) {
    int slot = chunk * AccChunk + i;
    auto tc = tDgCoordGrouped(0, slot);
    int row = int(get<0>(tc)), col = int(get<1>(tc));
    int logical = row * N + col;
    if (atomicCAS(&tmem_owner[logical], -1, tid) != -1) atomicAdd(&errors[2], 1);
    product[logical] = acc(0, i);

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
            state[bi][kb](j) = BF16(bf16_to_f32(state[bi][kb](j)) * decay[row] + acc(0, i));
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
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) return 2;
    int value = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--warmup")) warmup = value;
    else if (!std::strcmp(argv[i], "--iters")) iters = value;
    else return 2;
  }
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
  int *dao, *duo, *dto, *dso, *derr;
  CHECK_CUDA(cudaMalloc(&dkt, kt.size()*sizeof(BF16))); CHECK_CUDA(cudaMalloc(&du, u.size()*sizeof(BF16)));
  CHECK_CUDA(cudaMalloc(&di, initial.size()*sizeof(BF16))); CHECK_CUDA(cudaMalloc(&dg, decay.size()*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dprod, M*N*sizeof(float))); CHECK_CUDA(cudaMalloc(&dout, M*N*sizeof(BF16)));
  CHECK_CUDA(cudaMalloc(&dao, M*K*sizeof(int))); CHECK_CUDA(cudaMalloc(&duo, K*N*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dto, M*N*sizeof(int))); CHECK_CUDA(cudaMalloc(&dso, M*N*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&derr, 5*sizeof(int)));
  CHECK_CUDA(cudaMemcpy(dkt,kt.data(),kt.size()*sizeof(BF16),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(du,u.data(),u.size()*sizeof(BF16),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(di,initial.data(),initial.size()*sizeof(BF16),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dg,decay.data(),decay.size()*sizeof(float),cudaMemcpyHostToDevice));
  auto launch = [&] {
    CHECK_CUDA(cudaMemset(dao,0xff,M*K*sizeof(int))); CHECK_CUDA(cudaMemset(duo,0xff,K*N*sizeof(int)));
    CHECK_CUDA(cudaMemset(dto,0xff,M*N*sizeof(int))); CHECK_CUDA(cudaMemset(dso,0xff,M*N*sizeof(int)));
    CHECK_CUDA(cudaMemset(derr,0,5*sizeof(int)));
    v1b_mapping_kernel<<<1,NumThreads,sizeof(SharedStorage)>>>(dkt,du,di,dg,dprod,dout,dao,duo,dto,dso,derr);
  };
  for (int i=0;i<warmup;++i) launch(); CHECK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t a,b; CHECK_CUDA(cudaEventCreate(&a)); CHECK_CUDA(cudaEventCreate(&b)); CHECK_CUDA(cudaEventRecord(a));
  for (int i=0;i<iters;++i) launch(); CHECK_CUDA(cudaEventRecord(b)); CHECK_CUDA(cudaEventSynchronize(b));
  float ms; CHECK_CUDA(cudaEventElapsedTime(&ms,a,b));
  std::vector<float> product(M*N); std::vector<BF16> output(M*N);
  std::vector<int> ao(M*K),uo(K*N),to(M*N),so(M*N),errors(5);
  CHECK_CUDA(cudaMemcpy(product.data(),dprod,M*N*sizeof(float),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(output.data(),dout,M*N*sizeof(BF16),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ao.data(),dao,M*K*sizeof(int),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(uo.data(),duo,K*N*sizeof(int),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(to.data(),dto,M*N*sizeof(int),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(so.data(),dso,M*N*sizeof(int),cudaMemcpyDeviceToHost));
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
  bool direct = !owner_mismatch && !errors[3];
  bool correct = !missing_a&&!missing_u&&!missing_t&&!missing_s&&!errors[0]&&!errors[1]&&!errors[2]&&
                 !errors[4]&&!bad_product&&!bad_update&&direct;
  cudaFuncAttributes attr{}; CHECK_CUDA(cudaFuncGetAttributes(&attr,v1b_mapping_kernel));
  std::printf("{\"correct\":%s,\"mapping\":\"%s\",\"u_staging_bytes\":%zu,\"additional_smem_bytes\":%zu,"
    "\"tmem_columns\":%d,\"registers_per_thread\":%d,\"local_bytes_per_thread\":%zu,\"spills\":\"see_ptxas\","
    "\"v1b_mapping_go\":%s,\"missing_a\":%d,\"missing_u\":%d,"
    "\"missing_tmem\":%d,\"missing_state\":%d,\"owner_mismatch\":%d,\"duplicate_u\":%d,"
    "\"duplicate_tmem\":%d,\"duplicate_state\":%d,\"unmapped_local\":%d,\"bad_product\":%d,"
    "\"bad_bf16_update\":%d,\"max_product_error\":%.9g,\"mean_product_error\":%.9g,\"launch_us\":%.6f}\n",
    correct?"true":"false",direct?"DIRECT":"FULL_SMEM_REQUIRED",size_t(K*N*sizeof(BF16)),sizeof(SharedStorage),
    TmemColumns,attr.numRegs,attr.localSizeBytes,correct?"true":"false",missing_a,missing_u,missing_t,missing_s,
    owner_mismatch,errors[0],errors[2],errors[1],errors[3],
    bad_product,bad_update,max_error,sum_error/(M*N),ms*1000/iters);
  return correct?0:4;
}
