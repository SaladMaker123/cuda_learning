#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <mma.h>

#include "common/cuda_check.h"

namespace wmma = nvcuda::wmma;
constexpr int M = 256;
constexpr int N = 384;
constexpr int K = 256;
constexpr int BM = 16;
constexpr int BN = 48;
constexpr int BK = 16;
constexpr int STAGES = 2;
constexpr int RUNS = 30;
// 创建二维 TMA Tensor Map
CUtensorMap createTensorMap(
    __half* data,
    int width,
    int height,
    int tileWidth,
    int tileHeight
)
{
    CUtensorMap map{};
    cuuint64_t globalDim[2] = {(cuuint64_t)width,(cuuint64_t)height};
    cuuint64_t globalStrides[1] = {(cuuint64_t)(width * sizeof(__half))};
    cuuint32_t boxDim[2] = {(cuuint32_t)tileWidth,(cuuint32_t)tileHeight};
    cuuint32_t elementStrides[2] = {1,1};
    CUresult result = cuTensorMapEncodeTiled(
        &map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        data,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (result != CUDA_SUCCESS)
    {
        const char* error = nullptr;
        cuGetErrorString(result, &error);
        std::cerr << "Tensor Map error: "
                  << (error ? error : "Unknown error")
                  << std::endl;
        std::exit(1);
    }

    return map;
}
// TMA + WMMA Tensor Core + 双缓冲
__global__ void tmaWmmaGemmV11(
    const __grid_constant__ CUtensorMap mapA,
    const __grid_constant__ CUtensorMap mapB,
    float* C
)
{
    __shared__ __align__(128) __half sA[STAGES][BM][BK];
    __shared__ __align__(128) __half sB[STAGES][BK][BN];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> ready[STAGES];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> filled[STAGES];

    int startRow = blockIdx.y * BM;
    int startCol = blockIdx.x * BN;
    int warpId = threadIdx.x / 32;

    // 初始化 Barrier
    if(threadIdx.x == 0)
    {
        for(int s = 0; s < STAGES;++s)
        {
            init(&ready[s],97);
            init(&filled[s],97);
        }
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    // Warp 0：生产者
    if(warpId == 0)
    {
        if(threadIdx.x == 0)
        {
            for(int phase = 0;phase < K / BK; ++phase)
            {
                int current = phase % STAGES;
                //等待两个输入buffer都可以复用
                auto token = ready[current].arrive();
                ready[current].wait(cuda::std::move(token));
                //A：沿列方向移动
                cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                    sA[current],
                    &mapA,
                    phase * BK,
                    startRow,
                    filled[current]
                );
                // B：沿行方向移动
                cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                    sB[current],
                    &mapB,
                    startCol,
                    phase * BK,
                    filled[current]
                );
                // 两次 TMA 由同一个 filled Barrier 跟踪
                (void)cuda::device::barrier_arrive_tx(filled[current],1,sizeof(sA[current])+sizeof(sB[current]));
            }
        }
    }
    // Warp 1、2、3：Tensor Core 计算
    else
    {
        int warpCol = warpId -1;
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bFrag;
        wmma::fragment<wmma::accumulator,16,16,16,float> cFrag;
        wmma::fill_fragment(cFrag,0.0f);
        
        // 初始两个 Buffer 都是空闲的
        ready[0].arrive();
        ready[1].arrive();
        for(int phase = 0;phase < K / BK;++phase)
        {
            int current = phase % STAGES;
            // 等待 A 和 B 的当前 Tile 都搬运完成
            auto token = filled[current].arrive();
            filled[current].wait(cuda::std::move(token));
            //从shared memory 加载到wmma fragment
            wmma::load_matrix_sync(aFrag,&sA[current][0][0],BK);
            wmma::load_matrix_sync(bFrag,&sB[current][0][warpCol * 16],BN);
             // Tensor Core 矩阵乘加
            wmma::mma_sync(cFrag,aFrag,bFrag,cFrag);
            // 消费者用完当前 Buffer，通知生产者
            if(phase + STAGES < K / BK)
            {
                ready[current].arrive();
            }
        }
        // K 方向累加全部完成后才写回 C
        wmma::store_matrix_sync(C + startRow * N + startCol + warpCol * 16,cFrag,N,wmma::mem_row_major);
    }
}
int main()
{
    // Host 数据
    __half hA[M * K];
    __half hB[K * N];
    float hC[M * N];
    float expected[M * N];
    //初始化
     for (int i = 0; i < M * K; ++i)
    {
        hA[i] = __half(((i % 9) - 4) * 0.25f);
    }

    for (int i = 0; i < K * N; ++i)
    {
        hB[i] = __half(((i % 7) - 3) * 0.25f);
    }
    // CPU 参考 GEMM
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                sum += __half2float(hA[row * K + k]) *
                       __half2float(hB[k * N + col]);
            }

            expected[row * N + col] = sum;
        }
    }
    __half *dA = nullptr;
    __half *dB = nullptr;
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(hC)));
    CUDA_CHECK(cudaMemcpy(
        dA, hA, sizeof(hA), cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, hB, sizeof(hB), cudaMemcpyHostToDevice
    ));
    // A 的二维形状：M×K，TMA Tile：BM×BK
    CUtensorMap mapA = createTensorMap(dA,K,M,BK,BM);
    // B 的二维形状：K×N，TMA Tile：BK×BN
    CUtensorMap mapB = createTensorMap(dB,N,K,BN,BK);
    dim3 grid(N / BN,M / BM);
    dim3 block(128);
    // 正确性测试
    tmaWmmaGemmV11<<<grid, block>>>(mapA, mapB, dC);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        hC, dC, sizeof(hC), cudaMemcpyDeviceToHost
    ));
    float maxDiff = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float diff = std::fabs(hC[i] - expected[i]);

        if (diff > maxDiff)
        {
            maxDiff = diff;
        }
    }

    std::cout << "V11 TMA + WMMA: "
              << (maxDiff < 1e-3f ? "PASS" : "FAIL")
              << "  Max difference: " << maxDiff
              << std::endl;

    if (maxDiff >= 1e-3f)
    {
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        return 1;
    }
     // 预热
    for (int i = 0; i < 5; ++i)
    {
        tmaWmmaGemmV11<<<grid, block>>>(mapA, mapB, dC);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < RUNS; ++i)
    {
        tmaWmmaGemmV11<<<grid, block>>>(mapA, mapB, dC);
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed, start, stop
    ));

    std::cout << "Average time: "
              << elapsed * 1000.0f / RUNS
              << " us" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}


