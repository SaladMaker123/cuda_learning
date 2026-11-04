#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "common/cuda_check.h"

namespace wmma = nvcuda::wmma;

constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 64;
// ============================================
// 1. Multi-Block WMMA Kernel
// ============================================
__global__ void wmmaMultiBlock(
    const __half* A,
    const __half* B,
    float* C
)
{
    //每个block负责32X32的输出
    __shared__ __align__(32) __half sA[32 * K];
    __shared__ __align__(32) __half sB[K * 32];
    //当前block协作加载A,B
    for(int idx = threadIdx.x; idx < 32 * K;idx += blockDim.x)
    {
        sA[idx] = A[blockIdx.y * 32 * K + idx];
    }
    for(int idx = threadIdx.x; idx < 32 * K;idx += blockDim.x)
    {
        sB[idx] = B[blockIdx.x * 32 * K + idx];
    }
    __syncthreads();
    // ========================================
    // 2. 计算当前 Warp 的位置
    // ========================================
    int warpIdx = threadIdx.x / 32;
    int warpRow = warpIdx / 2;
    int warpCol = warpIdx % 2;
    // 当前 Warp 在完整 C 中的起点
    int row = blockIdx.y * 32 + warpRow * 16;
    int col = blockIdx.x * 32 + warpCol * 16;
     // ========================================
    // 3. 声明 Fragment
    // ========================================
     wmma::fragment<
        wmma::matrix_a,16,16,16,
        __half,wmma::row_major
    > a_frag;

    wmma::fragment<
        wmma::matrix_b,16,16,16,
        __half,wmma::col_major
    > b_frag;

    wmma::fragment<
        wmma::accumulator,16,16,16,float
    > c_frag;

    wmma::fill_fragment(c_frag,0.0f);
    // ========================================
    // 4. K Phase 累加
    // ========================================
    for (int phase = 0; phase < K / 16; ++phase)
    {
        int k0 = phase * 16;

        wmma::load_matrix_sync(
            a_frag,
            sA + warpRow * 16 * K + k0,
            K
        );

        wmma::load_matrix_sync(
            b_frag,
            sB + warpCol * 16 * K + k0,
            K
        );

        wmma::mma_sync(
            c_frag,
            a_frag,
            b_frag,
            c_frag
        );
    }
    // ========================================
    // 5. 写回完整矩阵 C
    // ========================================

    wmma::store_matrix_sync(
        C + row * N + col,
        c_frag,
        N,
        wmma::mem_row_major
    );
}
// ============================================
// 2. 主函数
// ============================================

int main()
{
    __half hA[M * K];
    __half hB[K * N];
    float hC[M * N];

    // 初始化 A：行主序
    for (int row = 0; row < M; ++row)
    {
        for (int k = 0; k < K; ++k)
        {
            hA[row * K + k] =
                __half((row + k) % 5 - 2);
        }
    }

    // 初始化 B：列主序
    for (int col = 0; col < N; ++col)
    {
        for (int k = 0; k < K; ++k)
        {
            hB[col * K + k] =
                __half((k + 2 * col) % 5 - 2);
        }
    }

    // 分配 GPU 内存
    __half* dA;
    __half* dB;
    float* dC;

    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(hC)));

    // CPU -> GPU
    CUDA_CHECK(cudaMemcpy(
        dA, hA, sizeof(hA),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, hB, sizeof(hB),
        cudaMemcpyHostToDevice
    ));

    // 16 个 Block，每个 Block 128 个线程
    dim3 grid(N / 32, M / 32);

    wmmaMultiBlock<<<grid,128>>>(dA,dB,dC);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU -> CPU
    CUDA_CHECK(cudaMemcpy(
        hC, dC, sizeof(hC),
        cudaMemcpyDeviceToHost
    ));

    // CPU 正确性验证
    int errors = 0;
    float maxDiff = 0.0f;

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float expected = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                expected +=
                    float(hA[row * K + k]) *
                    float(hB[col * K + k]);
            }

            float actual = hC[row * N + col];
            float diff = std::fabs(actual - expected);

            if (diff > maxDiff)
            {
                maxDiff = diff;
            }

            if (diff > 1e-3f)
            {
                ++errors;
            }
        }
    }

    std::cout << "Max difference: "
              << maxDiff << std::endl;

    if (errors == 0)
    {
        std::cout << "WMMA Multi-Block: PASS" << std::endl;
    }
    else
    {
        std::cout << "WMMA Multi-Block: FAIL ("
                  << errors << " errors)" << std::endl;
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return errors == 0 ? 0 : 1;
}