#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

#include "../../common/cuda_check.h"

//GPU缓存延迟测试kernel
__global__ void cacheLatencyKernel(
    const unsigned int* next,
    unsigned int start,
    int iterations,
    unsigned int* sink,
    bool useL1
)
{
    unsigned int idx = start;

    if (useL1)
    {
        // 使用 L1 + L2 Cache
        for (int i = 0; i < iterations; ++i)
        {
            idx = __ldca(&next[idx]);
        }
    }
    else
    {
        // 绕过 L1，使用 L2 及更低层次
        for (int i = 0; i < iterations; ++i)
        {
            idx = __ldcg(&next[idx]);
        }
    }

    sink[0] = idx;
}
//cpu:构造随机完整单环
std::vector<unsigned int> buildRandomCycle(int N)
{
    std::vector<unsigned int> order(N);
    std::vector<unsigned int> next(N);

    std::iota(order.begin(), order.end(), 0u);
    std::mt19937 rng(12345);
    std::shuffle(order.begin(), order.end(), rng);

    for(int i = 0; i<N;++i)
    {
        next[order[i]] = order[(i+1) % N];
    }
    return next;
}
// 测量指定工作集大小的平均访问延迟
void benchmarkLatency(int bytes, bool useL1)
{
    int N = bytes / sizeof(unsigned int);
    int iterations = std::max(1 << 20, N) + 17;
    //构造CPU数据
    std::vector<unsigned int> h_next = buildRandomCycle(N);
    
    unsigned int* d_next = nullptr;
    unsigned int* d_sink = nullptr;

    CUDA_CHECK(cudaMalloc(&d_next, bytes));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(unsigned int)));

    // 3. 复制数据到 GPU
    CUDA_CHECK(cudaMemcpy(
        d_next,
        h_next.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));
    //创建cudaEvent
    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    //预热
    for(int i = 0; i<2;++i)
    {
        cacheLatencyKernel<<<1,1>>>(d_next,0,iterations,d_sink,useL1);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    // 6. 重复测量三次
    std::vector<float> times;

    for(int r = 0;r < 3;++r)
    {
        CUDA_CHECK(cudaEventRecord(start));
        cacheLatencyKernel<<<1,1>>>(d_next,0,iterations,d_sink,useL1);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs,start,stop));
    times.push_back(elapsedMs);
    }
    // 7. 取中位数
    std::sort(times.begin(), times.end());
    float medianMs = times[1];
    double nsPerLoad =static_cast<double>(medianMs) * 1e6 / iterations;

    // 8. CPU 计算预期结果
    unsigned int expected = 0;

    for (int i = 0; i < iterations; ++i)
    {
        expected = h_next[expected];
    }
    unsigned int result = 0;
    CUDA_CHECK(cudaMemcpy(
        &result,
        d_sink,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    ));
    bool passed = (result == expected);
    std::cout << std::fixed << std::setprecision(2) 
            << "Mode = " << (useL1 ? "CA" : "CG")
                << " | "<< "Size = " << bytes / 1024 << " KB"
              << " | Time = " << medianMs << " ms"
              << " | Latency = " << nsPerLoad << " ns/load"
              << " | " << (passed ? "PASS" : "FAIL")
              << '\n';

    // 9. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_sink));

}
int main()
{
    for (int bytes : {
        32 * 1024,
        48 * 1024,
        64 * 1024,
        96 * 1024,
        128 * 1024,
        1024 * 1024,
        2 * 1024 * 1024,
        3 * 1024 * 1024,
        4 * 1024 * 1024
    })
    {
        benchmarkLatency(bytes, true);
        benchmarkLatency(bytes, false);
    }

    return 0;
}