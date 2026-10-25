#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "../common/cuda_check.h"


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


    // 每轮读取四个不同位置的元素
    for (int i = idx; i < N; i += 4 * stride)
    {
        sum0 += input[i];

        if (i + stride < N)
        {
            sum1 += input[i + stride];
        }
        if(i + 2 * stride < N)
        {
            sum2 += input[i + 2 * stride];
        }
         if (i + 3 * stride < N)
        {
            sum3 += input[i + 3 * stride];
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

    // 8. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_output));

    return passed ? avgMs : -1.0f;
}


int main()
{
    bool allPassed = true;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int activeBlocks = 0;

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocks,
        reducePartial,
        BLOCK_THREADS,
        0
    ));

    float occupancy =
        100.0f * activeBlocks * BLOCK_THREADS
        / prop.maxThreadsPerMultiProcessor;

    std::cout << "Active Blocks per SM: " << activeBlocks << '\n';
    std::cout << "Theoretical Occupancy: "
            << occupancy << "%\n";

    // 第一部分：原来的正确性测试
    for (int N : {1, 127, 128, 129, 1000, 1024, 10000, 100000})
    {
        bool passed = runTest(N);
        allPassed = passed && allPassed;
    }

    std::cout << "\n"
              << (allPassed ? "All tests PASSED!"
                            : "Some tests FAILED!")
              << '\n';

    if (!allPassed)
    {
        return 1;
    }

    // 第二部分：Benchmark
    constexpr int N = 1000000;

    int manyBlocks =
        (N + BLOCK_THREADS - 1) / BLOCK_THREADS;

    std::cout << "\nReduction Benchmark\n";
    float manyMs = benchmarkReduce(N, manyBlocks);
    float gridStrideMs = benchmarkReduce(N, 64);

    benchmarkReduce(N, 32);
    benchmarkReduce(N, 128);
    benchmarkReduce(N, 256);
    benchmarkReduce(N, 512);
    benchmarkReduce(N, 1024);

    if (manyMs <= 0.0f || gridStrideMs <= 0.0f)
    {
        return 1;
    }

    std::cout << "Time ratio (Many / 64) = "
              << manyMs / gridStrideMs
              << "x\n";

    return 0;
}