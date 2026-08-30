// 问题 4.3(FROM-SCRATCH,模块压轴):多级缓冲流水。
//
// 从你自己的 02_tma.cu 出发,把单缓冲扩成 STAGES 级循环缓冲:TMA 往
// 前预取后续 K 段,mma 消费当前段,装载与计算重叠。STAGES 是编译参数:
//   STAGES=4 make -B run/m4_gemm/03_pipeline
// (-B 不能省:只改 -D 不改文件,make 会认为无需重编。)
//
// 明确不要求:warp specialization、persistent kernel、epilogue 融合。
// 不设达成率门槛,评分看实验与归因质量。
//
// 两个已知事实,直接告知:
//   1. smem 用量 = STAGES*(BM+BN)*BK*2,STAGES>=3 起超过 48KB 静态
//      上限,必须动态 smem + cudaFuncSetAttribute(main 已配好)。
//   2. 一条真实的流水线 hazard(我们开发答案时踩到的,写出来让你避开):
//      "机会式预取"(try_wait 非阻塞,空了就发)不能替代"强制发射"。
//      若本轮要消费的那段 TMA 在早先检查时 stage 未空而被跳过,后面
//      wait full 等的就是一条从未发出的拷贝——死锁。症状签名很典型:
//      1024^3 侥幸全过,4096^3 必挂(13 万次机会必中一次)。正确结构:
//      本轮要消费的 TMA 用阻塞等 empty 保证发出,机会式 try_wait 只
//      用于更深的预取。另外 empty mbarrier 必须每 stage 一个:单个
//      mbar 的 parity 区分不了相隔 2 轮的完成,STAGES>=2 必然歧义。
//
// 交付:
//   - 梯子表第三行(4096^3,默认 STAGES=3)
//   - stages 扫描表:S ∈ {2,3,4,6},在两个形状上各扫一遍——4096^3 与
//     M=256 N=4096 K=16384(小 grid、长 K)。两张表的 S 敏感度不一样,
//     解释差异来自什么(提示方向:每 SM 常驻 block 数怎么随 smem 用量
//     变、块间并发本身能隐藏多少延迟)。./sweep_stages.sh 会跑全表
//   - 流水时空图:任选一个 S,画出稳态下 TMA/mma 在各 stage 上的重叠
//   - handout 4.3 的三问:瓶颈移动;梯子表逐级归因(含 assignment01
//     的 naive matmul 同口径对照);smem 与 TMEM 谁先顶住扩 stage/tile
//
// 运行:make run/m4_gemm/03_pipeline;./bin/m4_gemm/03_pipeline M N K
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

#ifndef STAGES
#define STAGES 3
#endif

constexpr int BM = 128, BN = 64, BK = 64;
constexpr int NSTAGE = STAGES;

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
}

__device__ inline bool mbar_try(
    uint32_t mbar,
    uint32_t phase) {
    uint32_t done = 0;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 "
        "p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}"
        : "=r"(done)
        : "r"(mbar), "r"(phase));
    return done != 0;
}


__global__ void gemm_pipeline(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                              float* gD, int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem =
        (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

    // TODO:把你 4.2 的 kernel 扩成 NSTAGE 级流水。参考结构:
    // (1) smem 划成 NSTAGE 段,stage s 的 A/B 起点自己排;mbarrier 每
    //     stage 两个:full[s](TMA 到达)、empty[s](mma 消费完成)
    // (2) 预热:先发 min(NSTAGE, iters) 轮 TMA(发第 it 轮 = 对 stage
    //     it%NSTAGE 做 arrive.expect_tx + 两条 cp.async.bulk.tensor)
    // (3) 主循环 it:
    //     - 强制发射:若第 it 轮 TMA 还没发,阻塞等 empty[it%NSTAGE]
    //       后补发(见文件头 hazard;empty 的 parity 按该 stage 被复用
    //       的轮次算,第一次复用等的是上一轮使用的完成)
    //     - 机会式深预取:try_wait 下一个待发 stage 的 empty,成功就
    //       继续发,失败立刻停,不许阻塞
    //     - 等 full[it%NSTAGE](parity = (it/NSTAGE)&1)→ tcgen05.fence
    //       → mma(与 4.2 相同,累加位口径不变)→ commit 到
    //       empty[it%NSTAGE]
    // (4) drain:等最后一轮 mma 的 empty 到达,再进 epilogue

      int tid = threadIdx.x;
      int warp = tid >> 5;
      int lane = tid & 31;
      
      constexpr uint32_t A_BYTES = BM * BK * sizeof(__nv_bfloat16);
      constexpr uint32_t B_BYTES = BN * BK * sizeof(__nv_bfloat16);
      constexpr uint32_t STAGE_BYTES = A_BYTES + B_BYTES;
      constexpr uint32_t TMA_BYTES = STAGE_BYTES;

      __shared__ uint32_t s_taddr[1];
      __shared__ __align__(8) uint64_t full[NSTAGE];
      __shared__  __align__(8) uint64_t empty[NSTAGE];

      uint32_t taddr_smem = (uint32_t)__cvta_generic_to_shared(s_taddr);

      constexpr int NUM_TMEM_COLS = BN;
     
      if (tid == 0) {
        for (int i = 0; i < NSTAGE; i++) {
            uint32_t full_addr = (uint32_t)__cvta_generic_to_shared(&full[i]);
            uint32_t empty_addr = (uint32_t)__cvta_generic_to_shared(&empty[i]);
            asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" ::"r"(full_addr),
                         "r"(1)
                         : "memory");
            asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" ::"r"(empty_addr), 
                         "r"(1)
                         : "memory");
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      
      if (warp == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;" ::
                "r"(taddr_smem),
            "r"(NUM_TMEM_COLS)
            : "memory");
      }

      __syncthreads();
      uint32_t taddr = s_taddr[0];

      int tileM = blockIdx.x * BM;
      int tileN = blockIdx.y * BN;

      int iters = K / BK;
      int warmup = min(iters,NSTAGE);

      for (int i = 0; i < warmup; i++) {
        int s = i % NSTAGE;
        uint8_t *stage_base = smem + s * STAGE_BYTES;
        uint8_t *sA = stage_base;
        uint8_t *sB = stage_base + A_BYTES;
        uint32_t full_addr = (uint32_t)__cvta_generic_to_shared(&full[s]);
        if (tid == 0) {
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :
                : "r"(full_addr),
                  "r"(TMA_BYTES)
                : "memory");
            uint32_t sA_addr =
                (uint32_t)__cvta_generic_to_shared(sA);

            uint64_t tmapA_addr =
                (uint64_t)&tmapA;

            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global."
                "mbarrier::complete_tx::bytes "
                "[%0], [%1, {%3, %4}], [%2];"
                :
                : "r"(sA_addr),
                  "l"(tmapA_addr),
                  "r"(full_addr),
                  "r"(i * BK),   // dim0 = K
                  "r"(tileM)      // dim1 = M
                : "memory");

            uint32_t sB_addr =
                (uint32_t)__cvta_generic_to_shared(sB);

            uint64_t tmapB_addr =
                (uint64_t)&tmapB;

            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global."
                "mbarrier::complete_tx::bytes "
                "[%0], [%1, {%3, %4}], [%2];"
                :
                : "r"(sB_addr),
                  "l"(tmapB_addr),
                  "r"(full_addr),
                  "r"(i * BK),   // dim0 = K
                  "r"(tileN)      // dim1 = N
                : "memory");

        }
    }
    
    int next_issue = warmup;

    for (int it = 0; it < K / BK; it++) {
        // 全体线程把 A 的 (tileM, it*BK) 块、B 的 (tileN, it*BK) 块
        // 按 swz128 布局 st.shared 进 smem(即 3.2 的 staging,行列起点换成 tile
        // 偏移) 由TMA 完成
        // issue four m128n64k16 tcgen05.mma
        
        int stage = it % NSTAGE;
        int gen = it / NSTAGE;
        uint8_t *stage_base = smem + stage * STAGE_BYTES;
        uint8_t *sA = stage_base;
        uint8_t *sB = stage_base + A_BYTES;
        
        uint32_t full_addr = (uint32_t)__cvta_generic_to_shared(&full[stage]);
        uint32_t empty_addr = (uint32_t)__cvta_generic_to_shared(&empty[stage]);
    
        if (tid == 0 && next_issue == it) {
            if (gen != 0) {
                mbar_wait(empty_addr,(uint32_t)((gen - 1) & 1));
            }
            uint8_t *issue_base = smem + stage * STAGE_BYTES;
            uint8_t *sA = issue_base;
            uint8_t *sB = issue_base + A_BYTES;
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :
                : "r"(full_addr),
                  "r"(TMA_BYTES)
                : "memory");
            uint32_t sA_addr =
                (uint32_t)__cvta_generic_to_shared(sA);

            uint64_t tmapA_addr =
                (uint64_t)&tmapA;

            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global."
                "mbarrier::complete_tx::bytes "
                "[%0], [%1, {%3, %4}], [%2];"
                :
                : "r"(sA_addr),
                  "l"(tmapA_addr),
                  "r"(full_addr),
                  "r"(it * BK),   // dim0 = K
                  "r"(tileM)      // dim1 = M
                : "memory");

            uint32_t sB_addr =
                (uint32_t)__cvta_generic_to_shared(sB);

            uint64_t tmapB_addr =
                (uint64_t)&tmapB;

            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global."
                "mbarrier::complete_tx::bytes "
                "[%0], [%1, {%3, %4}], [%2];"
                :
                : "r"(sB_addr),
                  "l"(tmapB_addr),
                  "r"(full_addr),
                  "r"(it * BK),   // dim0 = K
                  "r"(tileN)      // dim1 = N
                : "memory");
            next_issue++;             
        } 
    
        if (tid == 0) {
            while (next_issue < iters) {
                int p = next_issue;
                int p_stage = p % NSTAGE;
                int p_gen = p / NSTAGE;
                uint32_t p_empty_addr = (uint32_t)__cvta_generic_to_shared(&empty[p_stage]);

                if (!mbar_try(p_empty_addr,(uint32_t)((p_gen - 1) & 1))) {
                    break;
                }
                uint32_t p_full_addr = (uint32_t)__cvta_generic_to_shared(&full[p_stage]);
                
                uint8_t *p_base = smem + p_stage * STAGE_BYTES;
                uint8_t *pA = p_base;
                uint8_t *pB = p_base + A_BYTES;
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                    :
                    : "r"(p_full_addr),
                      "r"(TMA_BYTES)
                    : "memory");
                uint32_t pA_addr =
                    (uint32_t)__cvta_generic_to_shared(pA);
                uint64_t p_tmapA_addr =
                    (uint64_t)&tmapA;
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global."
                    "mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%3, %4}], [%2];"
                    :
                    : "r"(pA_addr),
                      "l"(p_tmapA_addr),
                      "r"(p_full_addr),
                      "r"(p * BK),   // dim0 = K
                      "r"(tileM)      // dim1 = M
                    : "memory");
                uint32_t pB_addr =
                    (uint32_t)__cvta_generic_to_shared(pB);
                uint64_t p_tmapB_addr =
                    (uint64_t)&tmapB;
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global."
                    "mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%3, %4}], [%2];"
                    :
                    : "r"(pB_addr),
                      "l"(p_tmapB_addr),
                      "r"(p_full_addr),
                      "r"(p * BK),   // dim0 = K
                      "r"(tileN)      // dim1 = N
                    : "memory");
                next_issue++;
            }
        }

        mbar_wait(full_addr,(uint32_t)(gen & 1));
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

        constexpr uint32_t IDESC = (1u << 4) | (1u << 7) | (1u << 10) |
                                   ((uint32_t)(BN >> 3) << 17) |
                                   ((uint32_t)(BM >> 4) << 24);

        if (tid == 0) {
          uint32_t sA_addr = (uint32_t)__cvta_generic_to_shared(sA);
          uint32_t sB_addr = (uint32_t)__cvta_generic_to_shared(sB);

          uint64_t descA0 = make_desc_sm100(sA_addr + 0, 0, 1024, 2);
          uint64_t descA1 = make_desc_sm100(sA_addr + 32, 0, 1024, 2);
          uint64_t descA2 = make_desc_sm100(sA_addr + 64, 0, 1024, 2);
          uint64_t descA3 = make_desc_sm100(sA_addr + 96, 0, 1024, 2);
          uint64_t descB0 = make_desc_sm100(sB_addr + 0, 0, 1024, 2);
          uint64_t descB1 = make_desc_sm100(sB_addr + 32, 0, 1024, 2);
          uint64_t descB2 = make_desc_sm100(sB_addr + 64, 0, 1024, 2);
          uint64_t descB3 = make_desc_sm100(sB_addr + 96, 0, 1024, 2);

          {
            uint32_t accumulate = (uint32_t)(!(it == 0));
            asm volatile("{\n"
                         "  .reg .pred p;\n"
                         "  setp.ne.u32 p, %4, 0;\n"
                         "  tcgen05.mma.cta_group::1.kind::f16 "
                         "    [%0], %1, %2, %3, p;\n"
                         "}\n"
                         :
                         : "r"(taddr), "l"(descA0), "l"(descB0), "r"(IDESC),
                           "r"(accumulate)
                         : "memory");
          }

          {
            uint32_t accumulate = 1;
            asm volatile("{\n"
                         "  .reg .pred p;\n"
                         "  setp.ne.u32 p, %4, 0;\n"
                         "  tcgen05.mma.cta_group::1.kind::f16 "
                         "    [%0], %1, %2, %3, p;\n"
                         "}\n"
                         :
                         : "r"(taddr), "l"(descA1), "l"(descB1), "r"(IDESC),
                           "r"(accumulate)
                         : "memory");
          }

          {
            uint32_t accumulate = 1;
            asm volatile("{\n"
                         "  .reg .pred p;\n"
                         "  setp.ne.u32 p, %4, 0;\n"
                         "  tcgen05.mma.cta_group::1.kind::f16 "
                         "    [%0], %1, %2, %3, p;\n"
                         "}\n"
                         :
                         : "r"(taddr), "l"(descA2), "l"(descB2), "r"(IDESC),
                           "r"(accumulate)
                         : "memory");
          }

          {
            uint32_t accumulate = 1;
            asm volatile("{\n"
                         "  .reg .pred p;\n"
                         "  setp.ne.u32 p, %4, 0;\n"
                         "  tcgen05.mma.cta_group::1.kind::f16 "
                         "    [%0], %1, %2, %3, p;\n"
                         "}\n"
                         :
                         : "r"(taddr), "l"(descA3), "l"(descB3), "r"(IDESC),
                           "r"(accumulate)
                         : "memory");
          }

          asm volatile(
              "tcgen05.commit.cta_group::1."
              "mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(empty_addr)
              : "memory");
        }

      }

      int num_it = K / BK;

      int last_it = num_it - 1;
      int last_stage = last_it % NSTAGE;
      int last_gen = last_it / NSTAGE;

      uint32_t last_empty_addr = (uint32_t)__cvta_generic_to_shared(&empty[last_stage]);

      mbar_wait(last_empty_addr,(uint32_t)(last_gen & 1));
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

      int row = warp * 32 + lane;

      for (int c = 0; c < BN; c += 8) {

        uint32_t src = taddr + ((uint32_t)(warp * 32) << 16) + c;

        float r[8];

        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]),
                       "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
                     : "r"(src));

        asm volatile("tcgen05.wait::ld.sync.aligned;");

#pragma unroll
        for (int i = 0; i < 8; i++) {
          gD[(tileM + row) * N + tileN + c + i] = r[i];
        }
      }

      __syncthreads();

      if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
            "r"(NUM_TMEM_COLS));
      }
}

int main(int argc, char** argv) {
    int M = argc > 3 ? atoi(argv[1]) : 4096;
    int N = argc > 3 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    if (M % BM || N % BN || K % BK) {
        printf("形状需按 %dx%dx%d 对齐\n", BM, BN, BK);
        return 1;
    }
    size_t nA = (size_t)M * K, nB = (size_t)N * K, nD = (size_t)M * N;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(nA), hB(nB);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    __nv_bfloat16 *dA, *dB;
    float *dD, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dB, nB * 2));
    CUDA_CHECK(cudaMalloc(&dD, nD * 4));
    CUDA_CHECK(cudaMalloc(&dRef, nD * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

    // TODO:tensor map 从你的 4.2 原样复制。
    alignas(64) CUtensorMap tmapA{};
    alignas(64) CUtensorMap tmapB{};

    cuuint64_t globalDimA[2] = {
      (cuuint64_t)K,
      (cuuint64_t)M
    };

    cuuint64_t globalStrideA[1] = {
      (cuuint64_t)K * sizeof(__nv_bfloat16)
    };

    cuuint32_t boxDimA[2] = {
      BK,
      BM
    };

    cuuint32_t elementStrideA[2] = {
      1, 1
    };

    CUresult retA = cuTensorMapEncodeTiled(
      &tmapA,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      dA,
      globalDimA,
      globalStrideA,
      boxDimA,
      elementStrideA,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    cuuint64_t globalDimB[2] = {
      (cuuint64_t)K,
      (cuuint64_t)N
    };

    cuuint64_t globalStrideB[1] = {
      (cuuint64_t)K * sizeof(__nv_bfloat16)
    };

    cuuint32_t boxDimB[2] = {
      BK,
      BN
    };

    cuuint32_t elementStrideB[2] = {
      1, 1
    };

    CUresult retB = cuTensorMapEncodeTiled(
      &tmapB,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      dB,
      globalDimB,
      globalStrideB,
      boxDimB,
      elementStrideB,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    dim3 grid(M / BM, N / BN);
    // NSTAGE=3 时 72KB+对齐余量,超 48KB 静态上限,动态 smem 必须。
    size_t smemBytes = (size_t)NSTAGE * (BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_pipeline,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smemBytes));
    auto launch = [&] {
        gemm_pipeline<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA,
                                                tmapB);
    };
    launch();
    CUDA_CHECK_KERNEL();

    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16BF,
                 K, dA, CUDA_R_16BF, K, &beta, dRef, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(nD), ref(nD);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, nD * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, nD * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < nD; i++) bad += got[i] != ref[i];

    int iters = (size_t)M * N >= (size_t)4096 * 4096 ? 20 : 100;
    float ms = time_avg_ms(launch, iters);
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    float cub_ms = time_avg_ms(
        [&] {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                         CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &beta, dRef,
                         CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
        },
        iters);
    double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);
    printf("[4.3 pipeline S=%d] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f "
           "TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
           NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops,
           cub_tflops, 100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
