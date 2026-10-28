
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../common/cuda_check.h"

// 实验配置
constexpr int kBlocks = 256;
constexpr int kThreads = 256;
constexpr int kWarmup = 3;
constexpr int kRepeat = 5;


// ======================================
// 1. CUDA Kernel
// ======================================
__global__ void cacheBandwidthKernel(
    const float4* __restrict__ input,
    float* __restrict__ sink,
    int M)
{
    int tid = threadIdx.x;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (int j = tid; j < M; j += blockDim.x)
    {
        float4 v = input[j];

        sum0 += v.x;
        sum1 += v.y;
        sum2 += v.z;
        sum3 += v.w;
    }

    float sum = (sum0 + sum1) + (sum2 + sum3);

    int outputIndex =
        blockIdx.x * blockDim.x + tid;

    sink[outputIndex] = sum;
}


// ======================================
// 2. Benchmark 函数
// ======================================
bool benchmarkBandwidth(size_t bytes)
{
    // float4 = 4 个 float = 16 字节
    int M = static_cast<int>(
        bytes / sizeof(float4));

    // CPU 输入：全部初始化为 1.0
    std::vector<float> h_input(
        bytes / sizeof(float), 1.0f);

    int outputCount = kBlocks * kThreads;

    std::vector<float> h_sink(outputCount, 0.0f);

    float4* d_input = nullptr;
    float* d_sink = nullptr;

    // GPU 显存分配
    CUDA_CHECK(cudaMalloc(&d_input, bytes));

    CUDA_CHECK(cudaMalloc(
        &d_sink,
        outputCount * sizeof(float)));

    // CPU → GPU
    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice));

    // 创建计时事件
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ==================================
    // 3. Warmup（预热）
    // ==================================
    for (int i = 0; i < kWarmup; ++i)
    {
        cacheBandwidthKernel<<<kBlocks, kThreads>>>(
            d_input, d_sink, M);

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================
    // 4. 正式计时
    // ==================================
    std::vector<float> times;

    for (int r = 0; r < kRepeat; ++r)
    {
        CUDA_CHECK(cudaEventRecord(start));

        cacheBandwidthKernel<<<kBlocks, kThreads>>>(
            d_input, d_sink, M);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsedMs = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsedMs, start, stop));

        times.push_back(elapsedMs);
    }

    // 取中位数
    std::sort(times.begin(), times.end());

    float medianMs = times[kRepeat / 2];

    // ==================================
    // 5. 读取输出并验证
    // ==================================
    CUDA_CHECK(cudaMemcpy(
        h_sink.data(),
        d_sink,
        outputCount * sizeof(float),
        cudaMemcpyDeviceToHost));

    bool pass = true;

    for (int i = 0; i < outputCount; ++i)
    {
        int tid = i % kThreads;

        int loads = 0;

        if (tid < M)
        {
            loads = (M - 1 - tid) / kThreads + 1;
        }

        float expected = static_cast<float>(
            loads * 4);

        if (!std::isfinite(h_sink[i]) ||
            std::fabs(h_sink[i] - expected) > 1e-3f)
        {
            pass = false;

            std::cout
                << "Mismatch at sink[" << i << "]"
                << ", expected = " << expected
                << ", actual = " << h_sink[i]
                << std::endl;

            break;
        }
    }

    // ==================================
    // 6. 计算有效读取带宽
    // ==================================
    double logicalBytes =
        static_cast<double>(bytes) * kBlocks;

    double seconds =
        static_cast<double>(medianMs) / 1000.0;

    double bandwidthGBs =
        logicalBytes / seconds / 1e9;

    std::cout
        << "Size = " << std::setw(5)
        << bytes / 1024
        << " KiB | Time = " << std::fixed
        << std::setprecision(3) << medianMs
        << " ms | BW = "
        << std::setprecision(2)
        << bandwidthGBs
        << " GB/s | "
        << (pass ? "PASS" : "FAIL")
        << std::endl;

    // ==================================
    // 7. 释放 CUDA 资源
    // ==================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_sink));

    return pass;
}


// ======================================
// 8. main
// ======================================
int main()
{
    for (size_t bytes : {
        size_t(32) * 1024,
        size_t(128) * 1024,
        size_t(1024) * 1024,
        size_t(2048) * 1024,
        size_t(4096) * 1024,
        size_t(16384) * 1024
    })
    {
        if (!benchmarkBandwidth(bytes))
        {
            std::cerr
                << "Benchmark failed."
                << std::endl;

            return 1;
        }
    }

    return 0;
}
