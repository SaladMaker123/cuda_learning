#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "../common/cuda_check.h"
#include <nvtx3/nvToolsExt.h>
#include <string>

constexpr int BLOCK_THREADS = 128;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int MAX_BLOCKS = 64;

//每个block计算一个部分和
__global__ void reducePartial(
    const float* input,
    float* partialOutput,
    int N
)
{
    __shared__ float partial[WARPS_PER_BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int warpId = tid / 32;
    int lane = tid % 32;

   int stride = gridDim.x * blockDim.x;

    // 四个独立的累加变量
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;


    // // 每轮读取四个不同位置的元素
    // for (int i = idx; i < N; i += 4 * stride)
    // {
    //     sum0 += input[i];

    //     if (i + stride < N)
    //     {
    //         sum1 += input[i + stride];
    //     }
    //     if(i + 2 * stride < N)
    //     {
    //         sum2 += input[i + 2 * stride];
    //     }
    //      if (i + 3 * stride < N)
    //     {
    //         sum3 += input[i + 3 * stride];
    //     }
    // }

    //通过向量化读取
    //计算完整float4的数量
    int N4 = N /4;
    //将输入指针转换为float4指针
    const float4* input4 = reinterpret_cast<const float4*>(input);
    //使用float4 进行向量化加载
    for(int i = idx;i < N4;i+=stride)
    {
        float4 v = input4[i];
        sum0 += v.x;
        sum1 += v.y;
        sum2 += v.z;
        sum3 += v.w;
    }
    //处理剩下的0~3个float
    if(idx == 0)
    {
        for(int j = N4 * 4; j < N;++j)
        {
            sum0 += input[j];
        }
    }
    //合并四条累加链
    float sum = (sum0 + sum1) + (sum2 + sum3);



    //第二阶段：每个warp内部完成归约
    for(int offset = 16;offset > 0;offset /=2)
    {
        sum += __shfl_down_sync(0xffffffffu,sum,offset);
    }
    // 第三阶段：每个 Warp 的 Lane 0 写入部分和
    if (lane == 0)
    {
        partial[warpId] = sum;
    }
    __syncthreads();
    // 第四阶段：Warp 0 汇总四个 Warp 的部分和

    if (warpId == 0)
    {
        float blockSum = (lane < WARPS_PER_BLOCK)
                    ? partial[lane] : 0.0f;

        for (int offset = 16; offset > 0; offset /= 2)
        {
            blockSum += __shfl_down_sync(
                0xffffffffu, blockSum, offset);
        }

        // 第五阶段：Warp 0 的 Lane 0 写回部分和
        if (lane == 0)
        {
            partialOutput[blockIdx.x] = blockSum;
        }
    }

}
// 最终归约：将所有 Block 的部分和合并
__global__ void reduceFinal(
    const float* partialInput,
    float* output,
    int numPartials)
{
    int lane = threadIdx.x;

    // 第一阶段：每个线程计算自己的局部和
    float sum = 0.0f;

    for (int i = lane; i < numPartials; i += warpSize)
    {
        sum += partialInput[i];
    }

    // 第二阶段：Warp Shuffle 归约
    for (int offset = 16; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(
            0xffffffffu, sum, offset);
    }

    // 第三阶段：写回最终结果
    if (lane == 0)
    {
        output[0] = sum;
    }
}
bool runTest(int N)
{
    // 1. CPU 准备输入和正确答案
    std::vector<float> h_input(N);
    double expected = 0.0;

    for (int i = 0; i < N; ++i)
    {
        h_input[i] = static_cast<float>(i + 1);
        expected += h_input[i];
    }
    // 2. 计算需要多少个 Block
    int numPartials = (N + BLOCK_THREADS - 1) / BLOCK_THREADS;
    if (numPartials > MAX_BLOCKS)
    {
        numPartials = MAX_BLOCKS;
    }

    size_t inputBytes = N * sizeof(float);
    size_t partialBytes = numPartials * sizeof(float);

    // 3. 分配 GPU 内存
    float* d_input = nullptr;
    float* d_partial = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, inputBytes));
    CUDA_CHECK(cudaMalloc(&d_partial, partialBytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    // 4. 将输入复制到 GPU
    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        inputBytes,
        cudaMemcpyHostToDevice
    ));

    // 5. 两级归约
    reducePartial<<<numPartials, BLOCK_THREADS>>>(
        d_input, d_partial, N
    );
    CUDA_CHECK(cudaGetLastError());

    reduceFinal<<<1, 32>>>(
        d_partial, d_output, numPartials
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. 复制最终结果回 CPU
    float gpuResult = 0.0f;
    CUDA_CHECK(cudaMemcpy(
        &gpuResult,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    // 7. 比较结果
    double error = std::fabs(gpuResult - expected);
    bool passed = error <= 1e-5 * expected;
      std::cout << "N = " << N
              << " | Blocks = " << numPartials
              << " | CPU = " << expected
              << " | GPU = " << gpuResult
              << " | " << (passed ? "PASS" : "FAIL")
              << '\n';

    // 8. 释放 GPU 内存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_output));

    return passed;
}
float benchmarkReduce(int N, int numBlocks)
{
    constexpr int WARMUP = 10;
    constexpr int REPEATS = 100;

    // 1. 准备测试数据：所有元素均为1
    std::vector<float> h_input(N, 1.0f);

    size_t inputBytes =
        static_cast<size_t>(N) * sizeof(float);

    size_t partialBytes =
        static_cast<size_t>(numBlocks) * sizeof(float);

    // 2. 分配GPU内存
    float* d_input = nullptr;
    float* d_partial = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, inputBytes));
    CUDA_CHECK(cudaMalloc(&d_partial, partialBytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        inputBytes,
        cudaMemcpyHostToDevice
    ));

    // 3. 创建CUDA Event
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 4. 预热
    for (int i = 0; i < WARMUP; ++i)
    {
        reducePartial<<<numBlocks, BLOCK_THREADS>>>(
            d_input, d_partial, N
        );
        CUDA_CHECK(cudaGetLastError());

        reduceFinal<<<1, 32>>>(
            d_partial, d_output, numBlocks
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. 开始计时
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; ++i)
    {
        reducePartial<<<numBlocks, BLOCK_THREADS>>>(
            d_input, d_partial, N
        );

        reduceFinal<<<1, 32>>>(
            d_partial, d_output, numBlocks
        );
    }

    CUDA_CHECK(cudaGetLastError());

    // 6. 结束计时
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsedMs, start, stop
    ));

    float avgMs = elapsedMs / REPEATS;
    // 额外测量：只运行 reducePartial
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; ++i)
    {
        reducePartial<<<numBlocks, BLOCK_THREADS>>>(
            d_input, d_partial, N
        );
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float partialMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&partialMs, start, stop));

    partialMs /= REPEATS;
    // 计算有效输入读取带宽
    double inputGB =
        static_cast<double>(N) * sizeof(float) / 1e9;

    double partialSeconds =
        static_cast<double>(partialMs) / 1000.0;

    double effectiveBandwidth =
        inputGB / partialSeconds;


    // 额外测量：只运行 reduceFinal
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; ++i)
    {
        reduceFinal<<<1, 32>>>(
            d_partial, d_output, numBlocks
        );
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float finalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&finalMs, start, stop));

    finalMs /= REPEATS;

    // 7. 验证最终结果
    float gpuResult = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &gpuResult,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool passed = (gpuResult == static_cast<float>(N));

    std::cout << "\nBlocks = " << numBlocks
            << " | GPU = " << gpuResult
            << " | " << (passed ? "PASS" : "FAIL")
            << '\n';

    std::cout << "Total Average:   " << avgMs << " ms\n";
    std::cout << "Partial Average: " << partialMs << " ms\n";
    std::cout << "Final Average:   " << finalMs << " ms\n";
    std::cout << "Effective Input BW: "<< effectiveBandwidth << " GB/s\n";

    // 8. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_output));

    return passed ? avgMs : -1.0f;
}


int main(int argc, char* argv[])
{
    bool allPassed = true;

    // 第一部分：正确性与边界测试
    for (int N : {
        1, 2, 3, 4, 5, 17,
        127, 128, 129,
        1000, 1003, 1024,
        10000, 100000, 1000003
    })
    {
        bool passed = runTest(N);
        allPassed = passed && allPassed;
    }

    if (!allPassed)
    {
        std::cout << "Correctness tests FAILED!\n";
        return 1;
    }

    std::cout << "\nAll correctness tests PASSED!\n";
    // 指定 --test-only 时，只运行正确性测试
    if (argc > 1 && std::string(argv[1]) == "--test-only")
    {
        return 0;
    }

    // 第二部分：固定 Block 数量，改变输入规模
    constexpr int NUM_BLOCKS = 256;

    std::cout << "\nReduction Bandwidth Benchmark\n";

    bool benchmarkPassed = true;
    for (int N : {1000000, 4000000, 16000000})
    {
        // 根据输入规模选择不同的 NVTX 名称
        const char* label =
            (N == 1000000) ? "Benchmark N=1M (4MB)" :
            (N == 4000000) ? "Benchmark N=4M (16MB)" :
                            "Benchmark N=16M (64MB)";

        // 开始 NVTX 范围
        nvtxRangePushA(label);

        // 执行原来的 Benchmark
        float ms = benchmarkReduce(N, NUM_BLOCKS);

        // 结束 NVTX 范围
        nvtxRangePop();

        if (ms <= 0.0f)
        {
            benchmarkPassed = false;
        }
    }


    return benchmarkPassed ? 0 : 1;
}