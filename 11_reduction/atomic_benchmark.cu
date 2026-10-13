#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_SIZE 256

__global__ void atomicReduction(
    const int* input,
    int* result,
    int N
)
{
    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N)
    {
        sdata[tid] = input[idx];
    }
    else
    {
        sdata[tid] = 0;
    }

    __syncthreads();

    //block内部树形规约
    for(int stride = blockDim.x/2;stride>0;stride/=2)
    {
        if(tid<stride)
        {
            sdata[tid] += sdata[tid+stride];
        }
        __syncthreads();
    }

    //每个block将部分原子和累加到最终结果
    if(tid == 0)
    {
        atomicAdd(result,sdata[0]);
    }
}
int main()
{
    const int N = 1024 * 1024;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int* input = nullptr;
    int* result = nullptr;

    // 分配输入数组
    cudaError_t err = cudaMallocManaged(
        &input, N * sizeof(int)
    );

    if (err != cudaSuccess)
    {
        printf("Input allocation failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    // 分配最终结果
    err = cudaMallocManaged(&result, sizeof(int));

    if (err != cudaSuccess)
    {
        printf("Result allocation failed: %s\n",
               cudaGetErrorString(err));
        cudaFree(input);
        return 1;
    }

    // 初始化输入数据
    for (int i = 0; i < N; i++)
    {
        input[i] = 1;
    }

    // 原子累加前必须清零
    result[0] = 0;

    // 启动 Atomic Reduction
    atomicReduction<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        input, result, N
    );

    // 检查 Kernel 启动
    err = cudaGetLastError();

    if (err == cudaSuccess)
    {
        err = cudaDeviceSynchronize();
    }

    if (err != cudaSuccess)
    {
        printf("Atomic kernel failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(input);
        cudaFree(result);
        return 1;
    }

    // 验证结果
    printf("Atomic final sum = %d\n", result[0]);
    printf("Expected sum = %d\n", N);

    if (result[0] != N)
    {
        printf("Result verification FAILED!\n");

        cudaFree(input);
        cudaFree(result);
        return 1;
    }

    printf("Result verification PASSED!\n");

    const int WARMUP = 5;
    const int REPEAT = 100;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    for (int i = 0; i < WARMUP; i++)
    {
        cudaMemsetAsync(result, 0, sizeof(int));

        atomicReduction<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            input, result, N
        );
    }

    // 等待预热完成
    cudaError_t warmupErr = cudaDeviceSynchronize();

    if (warmupErr != cudaSuccess)
    {
        printf("Warmup failed: %s\n",
            cudaGetErrorString(warmupErr));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(input);
        cudaFree(result);
        return 1;
    }

    // 开始计时
    cudaEventRecord(start);

    // 重复执行 100 次完整 Atomic Reduction
    for (int i = 0; i < REPEAT; i++)
    {
        // 每轮清零
        cudaMemsetAsync(result, 0, sizeof(int));

        // Atomic Reduction
        atomicReduction<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            input, result, N
        );
    }

    // 结束计时
    cudaEventRecord(stop);

    // 等待 GPU 完成
    cudaError_t timingErr = cudaEventSynchronize(stop);

    if (timingErr != cudaSuccess)
    {
        printf("Benchmark failed: %s\n",
            cudaGetErrorString(timingErr));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(input);
        cudaFree(result);
        return 1;
    }

    // 计算平均耗时
    float totalMs = 0.0f;
    cudaEventElapsedTime(&totalMs, start, stop);

    float avgMs = totalMs / REPEAT;

    printf("Atomic total time = %f ms\n", totalMs);
    printf("Atomic average time = %f ms\n", avgMs);

    // 验证重复测试结束后的结果
    printf("Benchmark final sum = %d\n", result[0]);

    if (result[0] != N)
    {
        printf("Benchmark verification FAILED!\n");

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(input);
        cudaFree(result);
        return 1;
    }

    printf("Benchmark verification PASSED!\n");

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(input);
    cudaFree(result);

    return 0;
}