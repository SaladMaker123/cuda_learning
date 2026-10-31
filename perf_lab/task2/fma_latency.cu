#include<cuda_runtime.h>
#include<cstdio>
#include<cmath>
#include<vector>
#include<algorithm>

#include"common/cuda_check.h"
// ============================================
// 1. 实验参数
// ============================================
constexpr int kBlocks = 1;
constexpr int kThreads = 1;

constexpr int kIterations = 200000;
constexpr int kWarmup = 2;
constexpr int kRepeat = 5;
constexpr int kMaxILP = 5;
// ============================================
// 2. 执行一次 FP32 FMA
// ============================================

__device__ __forceinline__ float executeFMA(
    float a,
    float b,
    float c)
{
    float result;

    asm volatile(
        "fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(a), "f"(b), "f"(c)
    );

    return result;
}
// ============================================
// 3. FMA Latency Kernel
// ============================================
template<int D,int I>
__global__ void fmaLatencyKernel(
    const float* input,
    float* output,
    int iterations
)
{
    //I个独立累加器
    float c[I];
    //读取初始值
    #pragma unroll
    for(int i = 0; i<I; i++)
    {
        c[i] = input[i];
    }
    //重复进行FMA计算
    for(int iter = 0; iter < iterations; ++iter)
    {
        //每条依赖链连续执行D次
        #pragma unroll
        for(int d = 0; d<D;d++)
        {
            //I条独立依赖链条
            #pragma unroll
            for(int i = 0;i < I;i++)
            {
                c[i] = executeFMA(1.0f,0.000001f,c[i]);
            }
        }
    }
    //保存结果
    #pragma unroll
    for(int i = 0;i < I;i++)
    {
        output[i] = c[i];
    }
}
// ============================================
// 4. 测试一组 D 和 ILP
// ============================================
template<int D,int I>
bool runCase(
    float* d_input,
    float* d_output,
    cudaEvent_t start,
    cudaEvent_t stop,
    int clockRateKHz
)
{
    //cpu端准备初始数据到GPU
    std::vector<float> h_input(I);
    for(int i = 0;i<I;i++)
    {
        h_input[i] = 1.0f + i; 
    }
    //复制初始数据到GPU
    CUDA_CHECK(cudaMemcpy(d_input,h_input.data(),I*sizeof(float),cudaMemcpyHostToDevice));
    //预热
    for(int i = 0;i< kWarmup;i++)
    {
        fmaLatencyKernel<D,I><<<kBlocks,kThreads>>>(d_input,d_output,kIterations);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    //正式测量
    std::vector<float> times;
    for(int r = 0;r<kRepeat;r++)
    {
        CUDA_CHECK(cudaEventRecord(start));
        fmaLatencyKernel<D,I><<<kBlocks,kThreads>>>(d_input,d_output,kIterations);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs,start,stop));
        times.push_back(elapsedMs);
    }
    //取中位数
    std::sort(times.begin(),times.end());
    float medianMs = times[times.size()/2];
    //每轮外层循环的估算时钟周期
    double cyclesPerIter = static_cast<double>(medianMs) * clockRateKHz / kIterations;
    // 一条依赖链每轮有 D 次 FMA
    // 仅作单链延迟的粗略参考
    double cyclesPerFMA = cyclesPerIter / D;
    //读取结果
    std::vector<float> h_output(I);
    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        I * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    // CPU 端计算预期结果
    std::vector<float> expected = h_input;
    for (int iter = 0; iter < kIterations; ++iter)
    {
        for (int d = 0; d < D; ++d)
        {
            for (int i = 0; i < I; ++i)
            {
                expected[i] = std::fma(
                    1.0f,
                    0.000001f,
                    expected[i]
                );
            }
        }
    }
    //验证结果
    bool passed = true;
    for(int i= 0; i<I;i++)
    {
        float difference = std::fabs(h_output[i] - expected[i]);
        if(difference > 0.01f)
        {
            passed =false;
        }
    }
     printf(
        "%3d %4d %14.2f %14.2f %10s\n",
        D,
        I,
        cyclesPerIter,
        cyclesPerFMA,
        passed ? "PASS" : "FAIL"
    );
    return passed;
}
// ============================================
// 5. 固定 D，测试不同 ILP
// ============================================
template <int D>
bool runILPSweep(
    float* d_input,
    float* d_output,
    cudaEvent_t start,
    cudaEvent_t stop,
    int clockRateKHz)
{
    bool passed = true;

    passed = runCase<D, 1>(
        d_input, d_output,
        start, stop, clockRateKHz
    ) && passed;

    passed = runCase<D, 2>(
        d_input, d_output,
        start, stop, clockRateKHz
    ) && passed;

    passed = runCase<D, 3>(
        d_input, d_output,
        start, stop, clockRateKHz
    ) && passed;

    passed = runCase<D, 4>(
        d_input, d_output,
        start, stop, clockRateKHz
    ) && passed;

    passed = runCase<D, 5>(
        d_input, d_output,
        start, stop, clockRateKHz
    ) && passed;

    return passed;
}
int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(
        &prop, 0
    ));

    printf("GPU: %s\n", prop.name);
    printf("Clock Rate: %d kHz\n\n", prop.clockRate);

    // GPU 显存指针
    float* d_input = nullptr;
    float* d_output = nullptr;

    // 分配 GPU 显存
    CUDA_CHECK(cudaMalloc(
        &d_input,
        kMaxILP * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_output,
        kMaxILP * sizeof(float)
    ));

    // 创建 CUDA Event
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("=== FMA Latency ===\n\n");

    printf(
        "%3s %4s %14s %14s %10s\n",
        "D",
        "ILP",
        "Cycles/iter",
        "Cycles/FMA",
        "Check"
    );

    bool allPassed = true;

    // D = 2
    allPassed = runILPSweep<2>(
        d_input, d_output,
        start, stop, prop.clockRate
    ) && allPassed;

    // D = 4
    allPassed = runILPSweep<4>(
        d_input, d_output,
        start, stop, prop.clockRate
    ) && allPassed;

    // D = 6
    allPassed = runILPSweep<6>(
        d_input, d_output,
        start, stop, prop.clockRate
    ) && allPassed;

    // D = 8
    allPassed = runILPSweep<8>(
        d_input, d_output,
        start, stop, prop.clockRate
    ) && allPassed;

    // D = 10
    allPassed = runILPSweep<10>(
        d_input, d_output,
        start, stop, prop.clockRate
    ) && allPassed;

    // 销毁 CUDA Event
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 释放 GPU 显存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    printf(
        "\nOverall: %s\n",
        allPassed ? "ALL PASS" : "FAIL"
    );

    return allPassed ? 0 : 1;
}