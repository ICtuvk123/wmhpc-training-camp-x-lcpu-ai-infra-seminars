// Isolate V1a's recurrence-boundary state movement. Production kernels and
// dispatch are intentionally not instantiated or modified by this probe.
#include "../../../csrc/smxx/utils.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using BF16 = cutlass::bfloat16_t;
using namespace cute;

constexpr int D = 128;
constexpr int Threads = 192;
constexpr int ComputeThreads = 128;

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1); \
} } while(0)

enum class BoundaryMode { Setup, Store, RoundTrip };

template <BoundaryMode Mode>
__global__ void boundary_kernel(BF16 const* input, BF16* output, unsigned* checksum) {
  if (threadIdx.x >= ComputeThreads) return;
  const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
  auto mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                            Layout<Shape<_1,_1>>{},Tile<_16,_16,_16>{});
  auto thr = mma.get_slice(lane);
  auto identity = make_identity_tensor(make_shape(Int<16>{},Int<16>{}));
  auto ref = make_tensor(make_gmem_ptr(static_cast<BF16*>(nullptr)),
                         make_layout(make_shape(Int<16>{},Int<16>{}),LayoutRight{}));
  using Frag = decltype(thr.partition_fragment_B(ref));
  Frag state[2][8];
  BF16 const* src = input + size_t(blockIdx.x)*D*D;
  BF16* dst = output + size_t(blockIdx.x)*D*D;
  unsigned coord_checksum = 0;

#pragma unroll
  for(int bi=0;bi<2;++bi) {
#pragma unroll
    for(int kb=0;kb<8;++kb) {
      auto coords=thr.partition_B(identity);
#pragma unroll
      for(int i=0;i<size(state[bi][kb]);++i) {
        auto coord=coords(i);
        int key=kb*16+int(get<1>(coord));
        int value=(warp*2+bi)*16+int(get<0>(coord));
        int logical=value*D+key;
        coord_checksum += unsigned(logical+1);
        if constexpr (Mode==BoundaryMode::RoundTrip) state[bi][kb](i)=src[logical];
        else if constexpr (Mode==BoundaryMode::Store)
          state[bi][kb](i)=BF16(float((logical%127)-63)/128.0f);
      }
      if constexpr (Mode!=BoundaryMode::Setup) {
        // Force the values to materialize as the same packed register fragment
        // used by V1a instead of allowing a direct GMEM-to-GMEM optimization.
        auto* words=reinterpret_cast<uint32_t*>(&state[bi][kb](0));
#pragma unroll
        for(int word=0;word<4;++word) asm volatile("" : "+r"(words[word]));
      }
    }
  }

  if constexpr (Mode==BoundaryMode::Setup) {
    checksum[size_t(blockIdx.x)*ComputeThreads+tid]=coord_checksum;
  } else {
#pragma unroll
    for(int bi=0;bi<2;++bi) {
#pragma unroll
      for(int kb=0;kb<8;++kb) {
        auto coords=thr.partition_B(identity);
#pragma unroll
        for(int i=0;i<size(state[bi][kb]);++i) {
          auto coord=coords(i);
          int key=kb*16+int(get<1>(coord));
          int value=(warp*2+bi)*16+int(get<0>(coord));
          dst[value*D+key]=state[bi][kb](i);
        }
      }
    }
  }
}

struct Variant {
  const char* name;
  void const* function;
  cudaFuncAttributes attrs{};
};

int main(int argc,char** argv) {
  int batch=4,heads=64,warmup=30,iters=200,rounds=5;
  for(int i=1;i<argc;i+=2) {
    if(i+1>=argc) return 2;
    int value=std::atoi(argv[i+1]);
    if(!std::strcmp(argv[i],"--batch")) batch=value;
    else if(!std::strcmp(argv[i],"--heads")) heads=value;
    else if(!std::strcmp(argv[i],"--warmup")) warmup=value;
    else if(!std::strcmp(argv[i],"--iters")) iters=value;
    else if(!std::strcmp(argv[i],"--rounds")) rounds=value;
    else return 2;
  }
  if(batch<1||heads<1||warmup<1||iters<1||rounds<1) return 2;
  int blocks=batch*heads;
  size_t elements=size_t(blocks)*D*D;
  BF16 *input=nullptr,*output=nullptr; unsigned* checksum=nullptr;
  CHECK(cudaMalloc(&input,elements*sizeof(BF16)));
  CHECK(cudaMalloc(&output,elements*sizeof(BF16)));
  CHECK(cudaMalloc(&checksum,size_t(blocks)*ComputeThreads*sizeof(unsigned)));
  std::vector<BF16> host(elements);
  for(size_t i=0;i<elements;++i) host[i]=BF16(float(int(i%127)-63)/128.0f);
  CHECK(cudaMemcpy(input,host.data(),elements*sizeof(BF16),cudaMemcpyHostToDevice));

  Variant variants[]={{"setup",reinterpret_cast<void const*>(boundary_kernel<BoundaryMode::Setup>)},
                      {"final_store",reinterpret_cast<void const*>(boundary_kernel<BoundaryMode::Store>)},
                      {"initial_load_plus_final_store",reinterpret_cast<void const*>(boundary_kernel<BoundaryMode::RoundTrip>)}};
  for(auto& v:variants) CHECK(cudaFuncGetAttributes(&v.attrs,v.function));
  cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
  cudaEvent_t start,stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
  double sums[3]={};
  auto launch=[&](int which) {
    if(which==0) boundary_kernel<BoundaryMode::Setup><<<blocks,Threads,0,stream>>>(input,output,checksum);
    if(which==1) boundary_kernel<BoundaryMode::Store><<<blocks,Threads,0,stream>>>(input,output,checksum);
    if(which==2) boundary_kernel<BoundaryMode::RoundTrip><<<blocks,Threads,0,stream>>>(input,output,checksum);
    CHECK(cudaGetLastError());
  };
  for(int round=0;round<rounds;++round) {
    for(int order=0;order<3;++order) {
      int which=(round%2==0)?order:2-order;
      for(int i=0;i<warmup;++i) launch(which);
      CHECK(cudaStreamSynchronize(stream));
      CHECK(cudaEventRecord(start,stream));
      for(int i=0;i<iters;++i) launch(which);
      CHECK(cudaEventRecord(stop,stream)); CHECK(cudaEventSynchronize(stop));
      float ms=0; CHECK(cudaEventElapsedTime(&ms,start,stop));
      double us=double(ms)*1000.0/iters; sums[which]+=us;
      auto const& v=variants[which];
      std::printf("{\"variant\":\"%s\",\"round\":%d,\"us\":%.6f,"
                  "\"registers_per_thread\":%d,\"local_bytes_per_thread\":%zu,"
                  "\"dynamic_smem_bytes\":0,\"blocks\":%d}\n",
                  v.name,round+1,us,v.attrs.numRegs,v.attrs.localSizeBytes,blocks);
    }
  }
  std::vector<BF16> actual(elements);
  launch(1); CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaMemcpy(actual.data(),output,elements*sizeof(BF16),cudaMemcpyDeviceToHost));
  bool store_correct=true;
  for(size_t i=0;i<elements;++i) {
    int logical=int(i%(D*D));
    BF16 expected(float((logical%127)-63)/128.0f);
    store_correct &= actual[i].storage==expected.storage;
  }
  launch(2); CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaMemcpy(actual.data(),output,elements*sizeof(BF16),cudaMemcpyDeviceToHost));
  bool roundtrip_correct=true;
  for(size_t i=0;i<elements;++i) roundtrip_correct &= actual[i].storage==host[i].storage;
  bool correct=store_correct&&roundtrip_correct;
  double setup=sums[0]/rounds,store=sums[1]/rounds,roundtrip=sums[2]/rounds;
  std::printf("{\"summary\":true,\"correct\":%s,\"store_correct\":%s,"
              "\"roundtrip_correct\":%s,\"setup_us\":%.6f,"
              "\"final_store_us\":%.6f,\"roundtrip_us\":%.6f,"
              "\"initial_load_increment_us\":%.6f}\n",
              correct?"true":"false",store_correct?"true":"false",
              roundtrip_correct?"true":"false",setup,store,roundtrip,roundtrip-store);
  cudaEventDestroy(start); cudaEventDestroy(stop); cudaStreamDestroy(stream);
  cudaFree(input); cudaFree(output); cudaFree(checksum);
  return correct?0:4;
}
