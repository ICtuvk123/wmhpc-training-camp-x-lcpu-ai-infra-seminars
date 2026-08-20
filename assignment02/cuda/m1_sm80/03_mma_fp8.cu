# include <cuda_fp8.h>
# include <cstdlib>
# include <random>
# include "../common.h"

// 实现一个单tile的fp8 mma,并判测是否 PASS。
// 使用m16n8k32d,f32累加.
__device__ __forceinline__
uint32_t pack4(uint8_t x0, uint8_t x1,
               uint8_t x2, uint8_t x3)
{
    return static_cast<uint32_t>(x0)
         | (static_cast<uint32_t>(x1) << 8)
         | (static_cast<uint32_t>(x2) << 16)
         | (static_cast<uint32_t>(x3) << 24);
}

__global__ void mma_fp8(__nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B, float *D) {
    int lane = threadIdx.x;
    int group = lane >> 2;
    int tig = lane & 3;
    
    const uint8_t *A_raw =
        reinterpret_cast<const uint8_t *>(A);

    const uint8_t *B_raw =
        reinterpret_cast<const uint8_t *>(B);
    unsigned ra[4];
    ra[0] = pack4(A_raw[group * 32 + tig * 4 + 0], A_raw[group * 32 + tig * 4 + 1], A_raw[group * 32 + tig * 4 + 2], A_raw[group * 32 + tig * 4 + 3]);
    ra[2] = pack4(A_raw[group * 32 + tig * 4 + 16], A_raw[group * 32 + tig * 4 + 17], A_raw[group * 32 + tig * 4 + 18], A_raw[group * 32 + tig * 4 + 19]);
    ra[2] = pack4(A_raw[(group + 8) * 32 + tig * 4 + 0], A_raw[(group + 8) * 32 + tig * 4 + 1], A_raw[(group + 8) * 32 + tig * 4 + 2], A_raw[(group + 8) * 32 + tig * 4 + 3]);
    ra[3] = pack4(A_raw[(group + 8) * 32 + tig * 4 + 16], A_raw[(group + 8) * 32 + tig * 4 + 17], A_raw[(group + 8) * 32 + tig * 4 + 18], A_raw[(group + 8) * 32 + tig * 4 + 19]);
    unsigned rb[2];
    rb[0] = pack4(B_raw[tig * 4 * 8 + group], B_raw[(tig * 4 + 1) * 8 + group], B_raw[(tig * 4 + 2) * 8 + group], B_raw[(tig * 4 + 3) * 8 + group]);
    rb[1] = pack4(B_raw[(tig * 4 + 16) * 8 + group], B_raw[(tig * 4 + 17) * 8 + group], B_raw[(tig * 4 + 18) * 8 + group], B_raw[(tig * 4 + 19) * 8 + group]);
    float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];

    asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, "
    "{%4,%5,%6,%7}, "
    "{%8,%9}, "
    "{%10,%11,%12,%13};\n"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
    : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
      "r"(rb[0]), "r"(rb[1]),
      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );

    D[group * 8 + tig * 2] = d[0];
    D[group * 8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

int main(int argc, char *argv[]) {
    int seed = argc > 1 ? atoi(argv[1]) : 0;

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-2, 2);
    float A_ref[16 * 32];
    float B_ref[32 * 8];

    for (int i = 0; i < 16 * 32; ++i)
        A_ref[i] = static_cast<float>(dist(gen));

    for (int i = 0; i < 32 * 8; ++i)
        B_ref[i] = static_cast<float>(dist(gen));
    
    __nv_fp8_e4m3 A_fp8[16 * 32];
    __nv_fp8_e4m3 B_fp8[32 * 8];

    for (int i = 0; i < 16 * 32; ++i)
        A_fp8[i] = __nv_fp8_e4m3(A_ref[i]);

    for (int i = 0; i < 32 * 8; ++i)
        B_fp8[i] = __nv_fp8_e4m3(B_ref[i]);

    float ref[16 * 8] = {};
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 32; k++)
                ref[r * 8 + n] +=
                    static_cast<float>(A_fp8[r * 32 + k]) *
                    static_cast<float>(B_fp8[k * 8 + n]);
    __nv_fp8_e4m3 *dA_fp8, *dB_fp8;

    float *dD;
    CUDA_CHECK(cudaMalloc(&dA_fp8, sizeof(A_fp8)));
    CUDA_CHECK(cudaMalloc(&dB_fp8, sizeof(B_fp8)));
    CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA_fp8, A_fp8, sizeof(A_fp8), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_fp8, B_fp8, sizeof(B_fp8), cudaMemcpyHostToDevice));
    mma_fp8<<<1, 32>>>(dA_fp8, dB_fp8, dD);
    CUDA_CHECK_KERNEL();
    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));

    long bad = 0;
    for (int r = 0; r < 16; r++) {
        for (int n = 0; n < 8; n++) {
            if (got[r * 8 + n] != ref[r * 8 + n]) {
                if (bad < 4)
                    printf("MISMATCH D[%d][%d]: got %.0f, want %.0f\n", r, n,
                           got[r * 8 + n], ref[r * 8 + n]);
                bad++;
            }
        }
    }
    if (bad)
        printf("FAIL: %ld / 128 mismatches\n", bad);
    else
        printf("PASS\n");
    return bad != 0;
}
