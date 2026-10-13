#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256

// ==========================================
// Kernel 1：计算每个 Block 的部分和
// ==========================================
__global__ void largeReduction(
    const float* input,
    float* partial,
    int N
)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 读取输入数据，并处理边界
    if (idx < N)
    {
        sdata[tid] = input[idx];
    }
    else
    {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    // Shared Memory 树形归约
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    // 每个 Block 输出一个部分和
    if (tid == 0)
    {
        partial[blockIdx.x] = sdata[0];
    }
}


// ==========================================
// Kernel 2：归约所有 Block 的部分和
// ==========================================
__global__ void finalReduction(
    const float* partial,
    float* result,
    int N
)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;

    // 每个线程累加多个部分和
    float localSum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x)
    {
        localSum += partial[i];
    }

    // 将局部和写入 Shared Memory
    sdata[tid] = localSum;

    __syncthreads();

    // Shared Memory 树形归约
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    // Thread 0 写出最终结果
    if (tid == 0)
    {
        result[0] = sdata[0];
    }
}


// ==========================================
// CPU 主函数
// ==========================================
int main()
{
    const int N = 1024 * 1024;
    const int NUM_BLOCKS =
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* input = nullptr;
    float* partial = nullptr;
    float* result = nullptr;

    cudaError_t err;

    // --------------------------------------
    // 1. 分配统一内存
    // --------------------------------------
    err = cudaMallocManaged(
        &input,
        N * sizeof(float)
    );

    if (err != cudaSuccess)
    {
        printf("Input allocation failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    err = cudaMallocManaged(
        &partial,
        NUM_BLOCKS * sizeof(float)
    );

    if (err != cudaSuccess)
    {
        printf("Partial allocation failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(input);
        return 1;
    }

    err = cudaMallocManaged(
        &result,
        sizeof(float)
    );

    if (err != cudaSuccess)
    {
        printf("Result allocation failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(input);
        cudaFree(partial);
        return 1;
    }

    // --------------------------------------
    // 2. 初始化 FP32 输入
    // --------------------------------------
    for (int i = 0; i < N; i++)
    {
        if (i % 2 == 0)
        {
            input[i] = 0.1f;
        }
        else
        {
            input[i] = -0.1f;
        }
    }

    // --------------------------------------
    // 3. 第一次 GPU Reduction
    // --------------------------------------
    largeReduction<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        input, partial, N
    );

    err = cudaGetLastError();

    if (err == cudaSuccess)
    {
        err = cudaDeviceSynchronize();
    }

    if (err != cudaSuccess)
    {
        printf("Kernel 1 failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(input);
        cudaFree(partial);
        cudaFree(result);
        return 1;
    }

    // --------------------------------------
    // 4. 第二次 GPU Reduction
    // --------------------------------------
    finalReduction<<<1, BLOCK_SIZE>>>(
        partial, result, NUM_BLOCKS
    );

    err = cudaGetLastError();

    if (err == cudaSuccess)
    {
        err = cudaDeviceSynchronize();
    }

    if (err != cudaSuccess)
    {
        printf("Kernel 2 failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(input);
        cudaFree(partial);
        cudaFree(result);
        return 1;
    }

    // --------------------------------------
    // 5. 输出前四个 Block 的部分和
    // --------------------------------------
    printf("===== GPU Partial Results =====\n");

    for (int i = 0; i < NUM_BLOCKS && i < 4; i++)
    {
        printf("partial[%d] = %.10f\n", i, partial[i]);
    }

    // --------------------------------------
    // 6. CPU 使用 float 累加部分和
    // --------------------------------------
    float cpuSum = 0.0f;

    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        cpuSum += partial[i];
    }

    // --------------------------------------
    // 7. CPU 使用 double 计算参考结果
    // --------------------------------------
    double referenceSum = 0.0;

    for (int i = 0; i < N; i++)
    {
        referenceSum += static_cast<double>(input[i]);
    }

    // --------------------------------------
    // 8. 计算 GPU 浮点误差
    // --------------------------------------
    double gpuSum = static_cast<double>(result[0]);

    double absError = fabs(gpuSum - referenceSum);

    double relativeError = 0.0;

    if (referenceSum != 0.0)
    {
        relativeError = absError / fabs(referenceSum);
    }

    // --------------------------------------
    // 9. 输出最终结果
    // --------------------------------------
    printf("\n===== FP32 Reduction Results =====\n");

    printf("N = %d\n", N);
    printf("Input value = %.10f\n", input[0]);

    printf("CPU float sum = %.10f\n", cpuSum);
    printf("GPU FP32 sum = %.10f\n", gpuSum);
    printf("Reference sum = %.10f\n", referenceSum);

    printf("Absolute error = %.10f\n", absError);
    printf("Relative error = %.12e\n", relativeError);

    // --------------------------------------
    // 10. 释放内存
    // --------------------------------------
    cudaFree(input);
    cudaFree(partial);
    cudaFree(result);

    return 0;
}