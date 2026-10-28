
#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../common/cuda_check.h"

// ======================================
// 1. 实验配置
// ======================================
constexpr int kBlocks = 256;
constexpr int kThreads = 256;

constexpr int kLoadsPerThread = 4096;
constexpr int kWarmup = 3;
constexpr int kRepeat = 5;


// ======================================
// 2. Shared Memory Bandwidth Kernel
// ======================================
__global__ void sharedBandwidthKernel(
    unsigned int* d_output,
    int vectorsPerThread,
    int repeat)
{
    int tid = threadIdx.x;

    // 动态共享内存
    extern __shared__ uint4 s_data[];

    // 每个线程初始化若干个 uint4
    for (int j = 0; j < vectorsPerThread; ++j)
    {
        int index = j * blockDim.x + tid;

        s_data[index] = make_uint4(1, 2, 3, 4);
    }

    // 等待 Block 内所有线程完成初始化
    __syncthreads();

    // 四条独立的累加链
    unsigned int sum0 = 0;
    unsigned int sum1 = 0;
    unsigned int sum2 = 0;
    unsigned int sum3 = 0;

    // 重复读取 Shared Memory
    for (int r = 0; r < repeat; ++r)
    {
        for (int j = 0; j < vectorsPerThread; ++j)
        {
            int index = j * blockDim.x + tid;

            // 防止编译器把重复加载直接移出循环
            asm volatile("" ::: "memory");

            // 从 Shared Memory 读取一个 uint4
            uint4 v = s_data[index];

            sum0 += v.x;
            sum1 += v.y;
            sum2 += v.z;
            sum3 += v.w;
        }
    }

    // 合并累加结果
    unsigned int sum =
        (sum0 + sum1) + (sum2 + sum3);

    // 每个线程写入一个独立输出位置
    int outputIndex = blockIdx.x * blockDim.x + tid;

    d_output[outputIndex] = sum;
}


// ======================================
// 3. 单组 Benchmark
// ======================================
bool benchmarkSharedBandwidth(int vectorsPerThread)
{
    // 当前 Block 需要多少字节 Shared Memory
    int sharedBytes =
        kThreads * vectorsPerThread * sizeof(uint4);

    // 每个线程重复多少轮
    int repeat = kLoadsPerThread / vectorsPerThread;

    // GPU 输出元素数量
    int outputCount = kBlocks * kThreads;

    // 每个线程的正确结果
    unsigned int expected = kLoadsPerThread * 10;

    // CPU 输出数组
    std::vector<unsigned int> h_output(outputCount, 0);

    // ----------------------------------
    // 第一步：分配 GPU 输出内存
    // ----------------------------------
    unsigned int* d_output = nullptr;

    size_t outputBytes =
        outputCount * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&d_output, outputBytes));

    // ----------------------------------
    // 第二步：Kernel 预热
    // ----------------------------------
    for (int i = 0; i < kWarmup; ++i)
    {
        sharedBandwidthKernel<<<
            kBlocks,
            kThreads,
            sharedBytes
        >>>(
            d_output,
            vectorsPerThread,
            repeat);

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ----------------------------------
    // 第三步：创建 CUDA Event
    // ----------------------------------
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ----------------------------------
    // 第四步：正式计时
    // ----------------------------------
    std::vector<float> times;

    for (int r = 0; r < kRepeat; ++r)
    {
        CUDA_CHECK(cudaEventRecord(start));

        sharedBandwidthKernel<<<
            kBlocks,
            kThreads,
            sharedBytes
        >>>(
            d_output,
            vectorsPerThread,
            repeat);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsedMs = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsedMs,
            start,
            stop));

        times.push_back(elapsedMs);
    }

    // ----------------------------------
    // 第五步：取中位数
    // ----------------------------------
    std::sort(times.begin(), times.end());

    float medianMs = times[kRepeat / 2];

    // ----------------------------------
    // 第六步：读取 GPU 输出
    // ----------------------------------
    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        outputBytes,
        cudaMemcpyDeviceToHost));

    // ----------------------------------
    // 第七步：CPU 正确性验证
    // ----------------------------------
    bool pass = true;

    for (int i = 0; i < outputCount; ++i)
    {
        if (h_output[i] != expected)
        {
            std::cout
                << "Mismatch at output[" << i << "]"
                << ", expected = " << expected
                << ", actual = " << h_output[i]
                << std::endl;

            pass = false;
            break;
        }
    }

    // ----------------------------------
    // 第八步：计算有效读取带宽
    // ----------------------------------
    double totalReadBytes =
        static_cast<double>(kBlocks) *
        kThreads *
        kLoadsPerThread *
        sizeof(uint4);

    double seconds = medianMs / 1000.0;

    double bandwidthGBps =
        totalReadBytes / seconds / 1e9;

    // ----------------------------------
    // 第九步：打印测试结果
    // ----------------------------------
    std::cout
        << "SMEM = " << std::setw(2)
        << sharedBytes / 1024
        << " KiB | Time = "
        << std::fixed << std::setprecision(3)
        << medianMs
        << " ms | BW = "
        << std::setprecision(2)
        << bandwidthGBps
        << " GB/s | "
        << (pass ? "PASS" : "FAIL")
        << std::endl;

    // ----------------------------------
    // 第十步：释放资源
    // ----------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_output));

    return pass;
}


// ======================================
// 4. 主函数
// ======================================
int main()
{
    // 每个线程在 Shared Memory 中
    // 分别使用 1、2、4、8 个 uint4
    int sizes[] = {1, 2, 4, 8};

    for (int i = 0; i < 4; ++i)
    {
        int vectorsPerThread = sizes[i];

        bool pass =
            benchmarkSharedBandwidth(vectorsPerThread);

        if (!pass)
        {
            std::cout << "Benchmark failed." << std::endl;
            return 1;
        }
    }

    return 0;
}
