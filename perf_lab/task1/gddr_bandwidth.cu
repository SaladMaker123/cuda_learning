
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>

#include "common/cuda_check.h"

// ==================== 基本配置 ====================

constexpr int kBlocks = 512;
constexpr int kThreads = 256;

constexpr int kWarmup = 3;
constexpr int kRepeat = 5;
// ==================== Kernel 1：初始化显存 ====================
__global__ void initializeKernel(float4* data, size_t N)
{
    // 计算当前线程的全局编号
    size_t tid = (size_t)blockIdx.x * blockDim.x
                 + threadIdx.x;

    // 整个 Grid 的线程总数
    size_t stride = (size_t)gridDim.x * blockDim.x;

    // 每个线程负责多个 float4 元素
    for (size_t i = tid; i < N; i += stride) {

        data[i] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
}
// ==================== Kernel 2：GDDR Bandwidth ====================
__global__ void gddrBandwidthKernel(
    const float4* input,
    float* output,
    int vectorPerThread
)
{
    //当前线程在BLOCK内的编号
    int tid = threadIdx.x;
    //每个block负责的float4数量
    int vectorsPerBlock = vectorPerThread * blockDim.x;
    // 当前block负责的数据起始位置
    size_t blockOffset = (size_t)blockIdx.x * vectorsPerBlock;
    //当前线程的累加器
    float sum = 0.0f;
    //每个线程读取多个float4
    for(int j = 0;j<vectorPerThread;j++)
    {
        size_t index = blockOffset + (size_t)j*blockDim.x + tid;
        //从global memory读取一个float4
        float4 v = input[index];
        //累加四个float分量
        sum += v.x + v.y + v.z + v.w;
    }
    //每个线程对应一个输出位置
    int outputIndex = blockIdx.x * blockDim.x + tid;
    // 保存当前线程的计算结果
    output[outputIndex] = sum;
}
// ==================== Host：主机端程序 ====================
int main()
{
    // 1. 准备测试规模：16 MiB ~ 4 GiB
    const int sizesMiB[] = {
        16, 24, 32, 40, 48, 56, 60, 64,
        72, 80, 96, 128, 256, 512,
        1024, 2048, 4096
    };
    const size_t maxBytes =
        4ULL * 1024 * 1024 * 1024;
    
    const int outputCount = kBlocks * kThreads;
     // 2. 检查 GPU 可用显存
    size_t freeBytes = 0;
    size_t totalDeviceBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(
        &freeBytes,&totalDeviceBytes
    ));
    printf("GPU Total Memory = %.2f GiB\n",
        (double)totalDeviceBytes / (1024.0 * 1024 * 1024));
    printf("GPU Free Memory  = %.2f GiB\n\n",
        (double)freeBytes / (1024.0 * 1024 * 1024));
    if (freeBytes < maxBytes) {
        printf("Not enough free GPU memory!\n");
        printf("Please close GPU applications and try again.\n");

        return 1;
    }
    // 3. 分配 GPU 显存

    float4* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, maxBytes));
    CUDA_CHECK(cudaMalloc(
        &d_output,
        (size_t)outputCount * sizeof(float)
    ));
    // 4. 初始化 Global Memory

    size_t totalVectors = maxBytes / sizeof(float4);

    initializeKernel<<<kBlocks, kThreads>>>(
        d_input,
        totalVectors
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // 5. 创建 CUDA Event

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> h_output(outputCount);

    bool allPassed = true;

    printf("=== GDDR Bandwidth ===\n\n");


    // 6. 遍历不同的测试规模

    for (int sizeMiB : sizesMiB) {

        size_t totalBytes =
            (size_t)sizeMiB * 1024 * 1024;

        // 计算每个线程需要读取多少个 float4
        int vectorsPerThread =
            (int)(totalBytes /
            (kBlocks * kThreads * sizeof(float4)));

        // 实际参与读取的总字节数
        size_t actualBytes =
            (size_t)kBlocks * kThreads *
            vectorsPerThread * sizeof(float4);


        // 7. Kernel 预热

        for (int i = 0; i < kWarmup; ++i) {

            gddrBandwidthKernel<<<kBlocks, kThreads>>>(
                d_input,
                d_output,
                vectorsPerThread
            );

            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaDeviceSynchronize());


        // 8. 重复测量

        std::vector<float> times;
        times.reserve(kRepeat);

        for (int i = 0; i < kRepeat; ++i) {

            CUDA_CHECK(cudaEventRecord(start));

            gddrBandwidthKernel<<<kBlocks, kThreads>>>(
                d_input,
                d_output,
                vectorsPerThread
            );

            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(stop));

            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsedMs = 0.0f;

            CUDA_CHECK(cudaEventElapsedTime(
                &elapsedMs,
                start,
                stop
            ));

            times.push_back(elapsedMs);
        }


        // 9. 取执行时间的中位数

        std::sort(times.begin(), times.end());

        float medianMs = times[kRepeat / 2];


        // 10. 计算有效读取带宽

        double seconds = medianMs / 1000.0;

        double bandwidthGBs =
            (double)actualBytes / seconds / 1e9;


        // 11. 将输出结果复制回 CPU

        CUDA_CHECK(cudaMemcpy(
            h_output.data(),
            d_output,
            (size_t)outputCount * sizeof(float),
            cudaMemcpyDeviceToHost
        ));


        // 12. 验证计算结果

        // 每个 float4 有四个 1.0f
        float expected = 4.0f * vectorsPerThread;

        bool passed = true;

        for (int i = 0; i < outputCount; ++i) {

            if (h_output[i] != expected) {

                passed = false;
                break;
            }
        }

        if (!passed) {
            allPassed = false;
        }


        // 13. 输出实验结果

        printf(
            "%5d MiB | Time = %8.3f ms | "
            "BW = %8.2f GB/s | %s\n",
            sizeMiB,
            medianMs,
            bandwidthGBs,
            passed ? "PASS" : "FAIL"
        );
    }


    // 14. 释放资源

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    printf("\nOverall: %s\n",
        allPassed ? "ALL PASS" : "FAIL");

    return allPassed ? 0 : 1;

}