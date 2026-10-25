#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../common/cuda_check.h"

__global__ void warpReduceSum(
    const float* input,
    float* output
)
{
    int lane = threadIdx.x;
    float sum = input[lane];
    for(int offset = 16;offset>0;offset /=2)
    {
        sum += __shfl_down_sync(0xffffffffu,sum,offset);
    }
    if(lane == 0)
    {
        output[0] = sum;
    }
}



__global__ void sharedReduceSum(
    const float* input,
    float* output
)
{
    __shared__ float data[32];
    int tid = threadIdx.x;
    data[tid] = input[tid];
    __syncthreads();
    for(int offset = 16;offset > 0;offset /=2)
    {
        if(tid < offset)
        {
            data[tid] += data[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[0] = data[0];
    }
}

__global__ void blockReduceSum(
    const float* input,
    float* output
)
{
    __shared__ float partial[4];
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int lane = tid % 32;
    //第一阶段：每个warp内部求和
    float sum = input[tid];
    for(int offset = 16;offset > 0;offset/=2)
    {
        sum += __shfl_down_sync(0xffffffffu,sum,offset);
    }
     // 每个 Warp 的 Lane 0 保存部分和
    if (lane == 0)
    {
        partial[warpId] = sum;
    }
    __syncthreads();
    //第二阶段：由warp0汇总四个部分和
    if(warpId == 0)
    {
        float blockSum = (lane < 4) ? partial[lane] : 0.0f;
        for(int offset = 16;offset > 0;offset/=2)
        {
            blockSum += __shfl_down_sync(0xffffffffu,blockSum,offset);
        }
        if(lane == 0)
        {
            output[0] = blockSum;
        }
    }
}

__global__ void multiBlockPartialSum(
    const float* input,
    float* partialOutput
)
{
    __shared__ float partial[4];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int warpId = tid / 32;
    int lane = tid % 32;
    //第一阶段：每个warp内部求和
    float sum = input[idx];
    for( int offset = 16;offset>0; offset /= 2)
    {
        sum += __shfl_down_sync(0xffffffffu,sum,offset);
    }
     // 每个 Warp 保存自己的部分和
    if (lane == 0)
    {
        partial[warpId] = sum;
    }
    __syncthreads();
     // 第二阶段：Warp 0 汇总本 Block 的四个部分和
    if (warpId == 0)
    {
        float blockSum = (lane < 4) ? partial[lane] : 0.0f;

        for (int offset = 16; offset > 0; offset /= 2)
        {
            blockSum += __shfl_down_sync(
                0xffffffffu, blockSum, offset);
        }

        if (lane == 0)
        {
            partialOutput[blockIdx.x] = blockSum;
        }
    }
}
__global__ void finalReduceSum(
    const float* partialInput,
    float* output)
{
    int lane = threadIdx.x;

    // 前8个线程读取部分和，其余线程使用0
    float sum = (lane < 8) ? partialInput[lane] : 0.0f;

    // 一个Warp内部完成最终归约
    for (int offset = 16; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(
            0xffffffffu, sum, offset);
    }

    // 只有Lane 0写回最终结果
    if (lane == 0)
    {
        output[0] = sum;
    }
}
bool runMultiBlockTest()
{
    constexpr int N = 1024;
    constexpr int BLOCK_THREADS = 128;
    constexpr int NUM_BLOCKS = N / BLOCK_THREADS;

    size_t inputBytes = static_cast<size_t>(N) * sizeof(float);
    size_t partialBytes = NUM_BLOCKS * sizeof(float);

    // 1. CPU 准备输入与正确答案
    std::vector<float> h_input(N);
    std::vector<float> expectedPartial(NUM_BLOCKS, 0.0f);
    std::vector<float> h_partial(NUM_BLOCKS);

    float expectedTotal = 0.0f;

    for (int i = 0; i < N; ++i)
    {
        h_input[i] = static_cast<float>(i + 1);

        expectedTotal += h_input[i];

        expectedPartial[i / BLOCK_THREADS] += h_input[i];
    }

    // 2. 分配 GPU 内存
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

    CUDA_CHECK(cudaMemset(d_partial, 0, partialBytes));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    // 3. 第一个 Kernel：1024 个元素归约为 8 个部分和
    multiBlockPartialSum<<<NUM_BLOCKS, BLOCK_THREADS>>>(
        d_input, d_partial
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. 第二个 Kernel：8 个部分和归约为 1 个结果
    finalReduceSum<<<1, 32>>>(d_partial, d_output);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. 将两个阶段的结果复制回 CPU
    CUDA_CHECK(cudaMemcpy(
        h_partial.data(),
        d_partial,
        partialBytes,
        cudaMemcpyDeviceToHost
    ));

    float gpuTotal = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &gpuTotal,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // 6. 验证每个 Block 的部分和
    bool partialPassed = (h_partial == expectedPartial);

    std::cout << "\nMulti-block Reduction Test\n";

    for (int i = 0; i < NUM_BLOCKS; ++i)
    {
        std::cout << "Block " << i
                  << " | GPU: " << h_partial[i]
                  << " | CPU: " << expectedPartial[i]
                  << " | "
                  << (h_partial[i] == expectedPartial[i] ? "PASS" : "FAIL")
                  << '\n';
    }

    // 7. 验证最终结果
    bool totalPassed = (gpuTotal == expectedTotal);

    std::cout << "CPU expected: " << expectedTotal << '\n';
    std::cout << "GPU result:   " << gpuTotal << '\n';
    std::cout << (totalPassed ? "Final PASS" : "Final FAIL") << '\n';

    // 8. 释放 GPU 内存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_output));

    return partialPassed && totalPassed;
}
bool runTest()
{
    constexpr int N = 32;
    size_t bytes = N * sizeof(float);
    //cpu准备输入数据与正确答案
    std::vector<float> h_input(N);
    float expected = 0.0f;

    for(int i = 0;i < N; i++)
    {
        h_input[i] = static_cast<float>(i+1);
        expected += h_input[i];
    }

    //分配GPU内存
    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    // 3. 将输入数据复制到 GPU
    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    // 4. 启动一个 Warp，完成归约
    warpReduceSum<<<1, 32>>>(d_input, d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 5. 将 GPU 结果复制回 CPU
    float gpuResult = 0.0f;
    CUDA_CHECK(cudaMemcpy(
        &gpuResult,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    // 6. 检查 Warp Shuffle 版本
    bool shufflePassed = (gpuResult == expected);

    // 7. 清空输出，运行 Shared Memory 版本
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    sharedReduceSum<<<1, 32>>>(d_input, d_output);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 8. 将 Shared Memory 版本的结果复制回 CPU
    float sharedResult = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &sharedResult,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool sharedPassed = (sharedResult == expected);

    // 9. 打印两种版本的结果
    std::cout << "CPU expected:  " << expected << '\n';

    std::cout << "Shuffle result: " << gpuResult
            << " | " << (shufflePassed ? "PASS" : "FAIL")
            << '\n';

    std::cout << "Shared result:  " << sharedResult
            << " | " << (sharedPassed ? "PASS" : "FAIL")
            << '\n';

    // 10. 释放 GPU 内存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return shufflePassed && sharedPassed;

}
bool runBlockTest()
{
    constexpr int N = 128;
    size_t bytes = N * sizeof(float);

    // CPU 准备输入和正确答案
    std::vector<float> h_input(N);
    float expected = 0.0f;

    for (int i = 0; i < N; ++i)
    {
        h_input[i] = static_cast<float>(i + 1);
        expected += h_input[i];
    }

    // GPU 内存分配
    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    // 启动 128 线程 Block Reduction
    blockReduceSum<<<1, 128>>>(d_input, d_output);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 读取结果
    float gpuResult = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &gpuResult,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    bool passed = (gpuResult == expected);

    std::cout << "\nBlock Reduction Test\n";
    std::cout << "CPU expected: " << expected << '\n';
    std::cout << "GPU result:   " << gpuResult << '\n';
    std::cout << (passed ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return passed;
}
int main()
{
    bool warpPassed = runTest();
    bool blockPassed = runBlockTest();
    bool multiPassed = runMultiBlockTest();

    bool allPassed = warpPassed && blockPassed && multiPassed;

    std::cout << "\n"
              << (allPassed ? "All tests PASSED!"
                            : "Some tests FAILED!")
              << std::endl;

    return allPassed ? 0 : 1;
}