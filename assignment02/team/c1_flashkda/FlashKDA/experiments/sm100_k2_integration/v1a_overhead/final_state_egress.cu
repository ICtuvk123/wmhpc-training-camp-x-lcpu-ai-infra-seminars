// Paired epilogue-only final-state egress diagnostics. No production changes.
#include "../../../csrc/smxx/fwd_kernel2.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using BF16=cutlass::bfloat16_t;
using namespace cute;
constexpr int D=128,Threads=192,ComputeThreads=128;
using Layouts=K2Layouts<D,16>;
using StateLayout=typename Layouts::StateSmemLayout;
using TMAStateLayout=typename Layouts::TMAStateSmemLayout;

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1); \
} } while(0)

enum class Mode { Empty, BaselinePrepare, BaselineTma, RegPrepare, Direct, Staged };

template<Mode mode,class TmaStore>
__global__ void egress_kernel(CUTE_GRID_CONSTANT TmaStore const tma_store,
                              BF16 const* input,BF16* output) {
  extern __shared__ __align__(128) BF16 storage[];
  Tensor s_state=make_tensor(make_smem_ptr(storage),StateLayout{});
  const int tid=int(threadIdx.x);
  if constexpr(mode==Mode::Empty) return;

  if constexpr(mode==Mode::BaselinePrepare||mode==Mode::BaselineTma) {
    BF16 const* src=input+size_t(blockIdx.x)*D*D;
    // This setup is reported separately. Volatile prevents dead-store removal
    // in the prepare-only control kernel.
    volatile uint16_t* smem=reinterpret_cast<volatile uint16_t*>(storage);
    for(int linear=tid;linear<D*D;linear+=Threads) {
      int row=linear/D,col=linear%D;
      smem[s_state.layout()(row,col)]=src[linear].storage;
    }
    if constexpr(mode==Mode::BaselinePrepare) {
      asm volatile("" ::: "memory");
      return;
    }
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
  }

  if constexpr(mode==Mode::RegPrepare||mode==Mode::Direct||mode==Mode::Staged) {
    if(tid<ComputeThreads) {
      int warp=tid/32,lane=tid%32;
      auto mma=make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                              Layout<Shape<_1,_1>>{},Tile<_16,_16,_16>{});
      auto thr=mma.get_slice(lane);
      auto identity=make_identity_tensor(make_shape(Int<16>{},Int<16>{}));
      auto ref=make_tensor(make_gmem_ptr(static_cast<BF16*>(nullptr)),
          make_layout(make_shape(Int<16>{},Int<16>{}),LayoutRight{}));
      using Frag=decltype(thr.partition_fragment_B(ref));
      Frag state[2][8];
      BF16 const* src=input+size_t(blockIdx.x)*D*D;
#pragma unroll
      for(int bi=0;bi<2;++bi) for(int kb=0;kb<8;++kb) {
        auto coords=thr.partition_B(identity);
#pragma unroll
        for(int i=0;i<size(state[bi][kb]);++i) {
          auto coord=coords(i);
          int key=kb*16+int(get<1>(coord));
          int value=(warp*2+bi)*16+int(get<0>(coord));
          state[bi][kb](i)=src[value*D+key];
        }
        auto* words=reinterpret_cast<uint32_t*>(&state[bi][kb](0));
#pragma unroll
        for(int word=0;word<4;++word) asm volatile("" : "+r"(words[word]));
      }
      if constexpr(mode!=Mode::RegPrepare) {
      BF16* dst=output+size_t(blockIdx.x)*D*D;
#pragma unroll
      for(int bi=0;bi<2;++bi) for(int kb=0;kb<8;++kb) {
        auto coords=thr.partition_B(identity);
#pragma unroll
        for(int i=0;i<size(state[bi][kb]);++i) {
          auto coord=coords(i);
          int key=kb*16+int(get<1>(coord));
          int value=(warp*2+bi)*16+int(get<0>(coord));
          if constexpr(mode==Mode::Direct) dst[value*D+key]=state[bi][kb](i);
          else s_state(value,key)=state[bi][kb](i);
        }
      }
      }
    }
    if constexpr(mode==Mode::RegPrepare) return;
    if constexpr(mode==Mode::Direct) return;
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
  }

  if constexpr(mode==Mode::BaselineTma||mode==Mode::Staged) {
    // Production assigns warp 4 to LOAD and warp 5 to STORE.
    if(tid==160) {
      Tensor g=tma_store.get_tma_tensor(make_shape(int(gridDim.x),Int<D>{},Int<D>{}));
      auto off=g.layout()(int(blockIdx.x),0,0);
      Tensor tile=make_tensor(g.data()+off,
          make_layout(make_shape(Int<1>{},Int<D>{},Int<D>{}),stride(g.layout())));
      auto cta=tma_store.get_slice(Int<0>{});
      cute::copy(tma_store,cta.partition_S(s_state),cta.partition_D(tile));
      tma_store_arrive();
      tma_store_wait<0>();
    }
  }
}

struct Result { const char* name; double sum=0; cudaFuncAttributes attr{}; };

void print_direct_store_audit() {
  int address[4][2][8][8][32] = {};
  int owned[128][128] = {};
  int owned_count[128] = {};
  for(int warp=0;warp<4;++warp) for(int lane=0;lane<32;++lane) {
    auto mma=make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                            Layout<Shape<_1,_1>>{},Tile<_16,_16,_16>{});
    auto thr=mma.get_slice(lane);
    auto identity=make_identity_tensor(make_shape(Int<16>{},Int<16>{}));
    auto coords=thr.partition_B(identity);
    int tid=warp*32+lane;
    for(int bi=0;bi<2;++bi) for(int kb=0;kb<8;++kb) for(int i=0;i<size(coords);++i) {
      auto coord=coords(i);
      int key=kb*16+int(get<1>(coord));
      int value=(warp*2+bi)*16+int(get<0>(coord));
      int logical=value*D+key;
      address[warp][bi][kb][i][lane]=logical;
      owned[tid][owned_count[tid]++]=logical;
    }
  }
  int total_runs=0,max_lane_group_span_bytes=0,adjacent_pairs=0;
  for(int warp=0;warp<4;++warp) for(int bi=0;bi<2;++bi)
    for(int kb=0;kb<8;++kb) for(int i=0;i<8;++i) {
      int sorted[32]; for(int lane=0;lane<32;++lane) sorted[lane]=address[warp][bi][kb][i][lane];
      std::sort(sorted,sorted+32); int runs=1;
      for(int lane=1;lane<32;++lane) runs+=sorted[lane]!=sorted[lane-1]+1;
      total_runs+=runs;
      max_lane_group_span_bytes=std::max(max_lane_group_span_bytes,2*(sorted[31]-sorted[0]+1));
    }
  for(int tid=0;tid<128;++tid) {
    std::sort(owned[tid],owned[tid]+owned_count[tid]);
    for(int i=1;i<owned_count[tid];++i) adjacent_pairs+=owned[tid][i]==owned[tid][i-1]+1;
  }
  std::printf("{\"direct_store_audit\":true,\"logical_bf16_values_per_compute_thread\":128,"
      "\"source_store_expression_bits\":16,\"lane_groups\":512,"
      "\"contiguous_runs_across_lane_groups\":%d,\"max_lane_group_span_bytes\":%d,"
      "\"same_thread_adjacent_value_pairs\":%d,"
      "\"sass_widths\":\"inspect STG instructions with cuobjdump\"}\n",
      total_runs,max_lane_group_span_bytes,adjacent_pairs);
}

int main(int argc,char** argv) {
  int batch=4,heads=64,warmup=30,iters=200,rounds=5;
  for(int i=1;i<argc;i+=2) {
    if(i+1>=argc) return 2; int v=std::atoi(argv[i+1]);
    if(!std::strcmp(argv[i],"--batch")) batch=v;
    else if(!std::strcmp(argv[i],"--heads")) heads=v;
    else if(!std::strcmp(argv[i],"--warmup")) warmup=v;
    else if(!std::strcmp(argv[i],"--iters")) iters=v;
    else if(!std::strcmp(argv[i],"--rounds")) rounds=v;
    else return 2;
  }
  print_direct_store_audit();
  int blocks=batch*heads; size_t count=size_t(blocks)*D*D;
  BF16 *input=nullptr,*output=nullptr;
  CHECK(cudaMalloc(&input,count*sizeof(BF16))); CHECK(cudaMalloc(&output,count*sizeof(BF16)));
  std::vector<BF16> host(count);
  for(size_t i=0;i<count;++i) host[i]=BF16(float(int(i%251)-125)/256.f);
  CHECK(cudaMemcpy(input,host.data(),count*sizeof(BF16),cudaMemcpyHostToDevice));
  auto g=make_tensor(make_gmem_ptr(output),make_layout(make_shape(blocks,D,D),LayoutRight{}));
  auto tma=make_tma_copy(SM90_TMA_STORE{},g,TMAStateLayout{});
  constexpr size_t smem_bytes=cosize_v<StateLayout>*sizeof(BF16);
  Result result[]={{"empty"},{"baseline_prepare_smem"},{"baseline_prepare_smem_tma"},
                   {"v1a_load_regs"},{"v1a_direct"},{"v1a_staged_tma"}};
#define ATTR(I,M) CHECK(cudaFuncGetAttributes(&result[I].attr,egress_kernel<Mode::M,decltype(tma)>))
  ATTR(0,Empty); ATTR(1,BaselinePrepare); ATTR(2,BaselineTma); ATTR(3,RegPrepare); ATTR(4,Direct); ATTR(5,Staged);
#undef ATTR
  cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
  cudaEvent_t start,stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
  auto launch=[&](int which) {
    switch(which) {
#define CASE(I,M) case I: egress_kernel<Mode::M><<<blocks,Threads,smem_bytes,stream>>>(tma,input,output); break
      CASE(0,Empty); CASE(1,BaselinePrepare); CASE(2,BaselineTma);
      CASE(3,RegPrepare); CASE(4,Direct); CASE(5,Staged);
#undef CASE
    }
    CHECK(cudaGetLastError());
  };
  // Check all three egress paths before timing.
  bool correct=true; std::vector<BF16> actual(count);
  for(int which:{2,4,5}) {
    CHECK(cudaMemset(output,0xff,count*sizeof(BF16))); launch(which); CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(actual.data(),output,count*sizeof(BF16),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<count;++i) correct &= actual[i].storage==host[i].storage;
  }
  for(int round=0;round<rounds;++round) for(int order=0;order<6;++order) {
    int which=(round&1)?5-order:order;
    for(int i=0;i<warmup;++i) launch(which); CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaEventRecord(start,stream)); for(int i=0;i<iters;++i) launch(which);
    CHECK(cudaEventRecord(stop,stream)); CHECK(cudaEventSynchronize(stop));
    float ms=0; CHECK(cudaEventElapsedTime(&ms,start,stop)); double us=ms*1000.0/iters;
    result[which].sum+=us;
    std::printf("{\"variant\":\"%s\",\"round\":%d,\"us\":%.6f,"
                "\"registers_per_thread\":%d,\"local_bytes_per_thread\":%zu,"
                "\"dynamic_smem_bytes\":%zu,\"spills\":\"see_ptxas\"}\n",
                result[which].name,round+1,us,result[which].attr.numRegs,
                result[which].attr.localSizeBytes,smem_bytes);
  }
  double mean[6]; for(int i=0;i<6;++i) mean[i]=result[i].sum/rounds;
  double launch_us=mean[0],baseline=mean[2]-mean[1],direct=mean[4]-mean[3],staged=mean[5]-mean[3];
  double recovered=mean[4]-mean[5];
  const char* decision=recovered>=8.0?"GO":recovered>=2.0?"LOW_ROI":"NO_GAIN";
  std::printf("{\"summary\":true,\"correct\":%s,\"launch_us\":%.6f,"
      "\"baseline_prepare_us\":%.6f,\"v1a_load_regs_us\":%.6f,"
      "\"baseline_final_store_us\":%.6f,\"v1a_direct_final_store_us\":%.6f,"
      "\"v1a_staged_final_store_us\":%.6f,\"v1a_direct_minus_baseline_us\":%.6f,"
      "\"v1a_staged_minus_baseline_us\":%.6f,\"staged_vs_direct_speedup\":%.6f,"
      "\"recovered_us\":%.6f,\"staging_smem_bytes\":%zu,\"barriers\":1,"
      "\"FINAL_EGRESS_OPTIMIZATION\":\"%s\"}\n",
      correct?"true":"false",launch_us,mean[1]-launch_us,mean[3]-launch_us,baseline,direct,staged,
      direct-baseline,staged-baseline,mean[4]/mean[5],recovered,smem_bytes,decision);
  cudaEventDestroy(start); cudaEventDestroy(stop); cudaStreamDestroy(stream);
  cudaFree(input); cudaFree(output); return correct?0:4;
}
