
#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../common/cuda_check.h"

// ======================================
// 1. 实验配置
// ======================================
constexpr int kIterations = (1 << 20) + 17;
constexpr int kWarmup = 2;
constexpr int kRepeat = 5;


// ======================================
// 2. Shared Memory Latency Kernel
// ======================================
__global__ void sharedLatencyKernel(
    const unsigned int* d_next,
    int N,
    int iterations,
    unsigned int* d_result,
    unsigned long long* d_cycles)
{
    // 动态共享内存数组
    extern __shared__ unsigned int s_next[];

    // 将数据从 Global Memory 复制到 Shared Memory
    // 初始化阶段不参与计时
    for (int i = 0; i < N; ++i)
    {
        s_next[i] = d_next[i];
    }

    // 保留真实的 Shared Memory 读取
    volatile unsigned int* p = s_next;

    // Pointer Chasing 从下标 0 开始
    unsigned int idx = 0;

    // 开始计时
    unsigned long long start = clock64();

    // 连续依赖加载
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        idx = p[idx];
    }

    // 结束计时
    unsigned long long stop = clock64();

    // 保存最终结果和执行周期数
    d_result[0] = idx;
    d_cycles[0] = stop - start;
}


// ======================================
// 3. 单组 Benchmark
// ======================================
bool benchmarkSharedLatency(int N)
{
    // 每个 unsigned int 占 4 字节
    int bytes = N * sizeof(unsigned int);

    // ==================================
    // 4. CPU 构造 Pointer Chasing 单环
    // ==================================
    std::vector<unsigned int> h_next(N);

    // 本实验 N 均为 2 的整数次幂
    // stride 为奇数，能够形成完整单环
    int stride = N / 2 + 1;

    for (int i = 0; i < N; ++i)
    {
        h_next[i] = (i + stride) % N;
    }

    // ==================================
    // 5. CPU 计算正确答案
    // ==================================
    unsigned int expected = 0;

    for (int i = 0; i < kIterations; ++i)
    {
        expected = h_next[expected];
    }

    // ==================================
    // 6. 分配 GPU 内存
    // ==================================
    unsigned int* d_next = nullptr;
    unsigned int* d_result = nullptr;
    unsigned long long* d_cycles = nullptr;

    CUDA_CHECK(cudaMalloc(&d_next, bytes));

    CUDA_CHECK(cudaMalloc(
        &d_result,
        sizeof(unsigned int)));

    CUDA_CHECK(cudaMalloc(
        &d_cycles,
        sizeof(unsigned long long)));

    // CPU -> GPU
    CUDA_CHECK(cudaMemcpy(
        d_next,
        h_next.data(),
        bytes,
        cudaMemcpyHostToDevice));

    // ==================================
    // 7. Kernel 预热
    // ==================================
    for (int i = 0; i < kWarmup; ++i)
    {
        sharedLatencyKernel<<<1, 1, bytes>>>(
            d_next,
            N,
            kIterations,
            d_result,
            d_cycles);

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================
    // 8. 正式测量
    // ==================================
    std::vector<unsigned long long> times;

    for (int r = 0; r < kRepeat; ++r)
    {
        sharedLatencyKernel<<<1, 1, bytes>>>(
            d_next,
            N,
            kIterations,
            d_result,
            d_cycles);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 读取本次 Kernel 记录的时钟周期数
        unsigned long long elapsedCycles = 0;

        CUDA_CHECK(cudaMemcpy(
            &elapsedCycles,
            d_cycles,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost));

        times.push_back(elapsedCycles);
    }

    // ==================================
    // 9. 取五次测量的中位数
    // ==================================
    std::sort(times.begin(), times.end());

    unsigned long long medianCycles =
        times[kRepeat / 2];

    // ==================================
    // 10. CPU 验证 GPU 结果
    // ==================================
    unsigned int actual = 0;

    CUDA_CHECK(cudaMemcpy(
        &actual,
        d_result,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost));

    bool pass = (actual == expected);

    // ==================================
    // 11. 计算平均依赖加载周期数
    // ==================================
    double cyclesPerLoad =
        static_cast<double>(medianCycles) / kIterations;

    std::cout
        << "Size = " << std::setw(5)
        << bytes
        << " B | Cycles/load = "
        << std::fixed
        << std::setprecision(2)
        << cyclesPerLoad
        << " | "
        << (pass ? "PASS" : "FAIL")
        << std::endl;

    if (!pass)
    {
        std::cout
            << "Expected = " << expected
            << ", Actual = " << actual
            << std::endl;
    }

    // ==================================
    // 12. 释放 GPU 资源
    // ==================================
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_cycles));

    return pass;
}


// ======================================
// 13. 主函数
// ======================================
int main()
{
    // Shared Memory 工作集大小
    int sizes[] = {
        32,
        256,
        1024,
        4096,
        8192
    };

    for (int N : sizes)
    {
        if (!benchmarkSharedLatency(N))
        {
            std::cerr << "Benchmark failed." << std::endl;
            return 1;
        }
    }

    return 0;
}
