#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_SIZE 256

__global__ void largeReduction(
    const int* input,
    int* partial,
    int N
)
{
    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;//block内部线程编号
    int idx = blockIdx.x * blockDim.x + threadIdx.x;//全局线程索引

    //加载数据，并处理数组边界
    if(idx<N)
    {
        sdata[tid] = input[idx];
    }
    else{
        sdata[tid] = 0;
    }
    __syncthreads();

    //block内部树形规约
    for(int stride = blockDim.x/2;stride > 0;stride/=2)
    {
        if(tid < stride)
        {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        __syncthreads();
    }

    //每个block输出一个部分和
    if(tid == 0)
    {
        partial[blockIdx.x] = sdata[0];
    }
}

//第二次规约kernel
 
__global__ void finalReduction(
    const int* partial,
    int* result,
    int N
)
{

    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    //每个线程计算自己的局部和
    int localSum = 0;
    for(int i = tid;i< N;i+=blockDim.x)
    {
        localSum += partial[i];
    }
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

int main()
{
    const int N = 1024 * 1024;
    //向上取整，计算需要的block数量
    const int NUM_BLOCKS = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
    
    int* input = nullptr;
    int* partial = nullptr;
    int* result = nullptr;

    //分配统一内存
    cudaError_t err = cudaMallocManaged(&input,N*sizeof(int));

    if (err != cudaSuccess)
    {
        printf("Input allocation failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMallocManaged(
        &partial,
        NUM_BLOCKS * sizeof(int)
    );

    if (err != cudaSuccess)
    {
        printf("Partial allocation failed: %s\n",
               cudaGetErrorString(err));
        cudaFree(input);
        return 1;
    }
    err = cudaMallocManaged(&result, sizeof(int));

    if (err != cudaSuccess)
    {
        printf("Result allocation failed: %s\n",
            cudaGetErrorString(err));

        cudaFree(input);
        cudaFree(partial);

        return 1;
    }

    // 初始化输入数据
    for (int i = 0; i < N; i++)
    {
        input[i] = 1;
    }

    // 启动第一阶段 Reduction
    largeReduction<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        input, partial, N
    );

    // 检查 Kernel 启动错误
    err = cudaGetLastError();

    if (err == cudaSuccess)
    {
        err = cudaDeviceSynchronize();
    }

    if (err != cudaSuccess)
    {
        printf("Kernel failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(input);
        cudaFree(partial);
        return 1;
    }

    //输出每个block的部分和
    for(int i = 0;i < 4;i++)
    {
        printf("partial[%d] = %d\n",i,partial[i]);
    }

    // CPU 计算部分和的总和，用于验证
    int cpuSum = 0;

    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        cpuSum += partial[i];
    }

    printf("CPU check sum = %d\n", cpuSum);

    // 第二阶段：GPU 归约 partial 数组
    finalReduction<<<1, BLOCK_SIZE>>>(
        partial, result, NUM_BLOCKS
    );

    // 检查第二个 Kernel 的启动错误
    err = cudaGetLastError();

    if (err == cudaSuccess)
    {
        err = cudaDeviceSynchronize();
    }

    if (err != cudaSuccess)
    {
        printf("Final reduction failed: %s\n",
            cudaGetErrorString(err));

        cudaFree(input);
        cudaFree(partial);
        cudaFree(result);

        return 1;
    }

    // 输出 GPU 最终结果
    printf("GPU Final sum = %d\n", result[0]);

    //创建CUDA EVENT变量
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //预热次数
    const int WARMUP = 5;
    //GPU预热
    for(int i = 0; i<WARMUP;i++)
    {
        largeReduction<<<NUM_BLOCKS,BLOCK_SIZE>>>(
            input,partial,N
        );  
    }
    // 等待预热结束
    cudaError_t warmupErr = cudaDeviceSynchronize();

    if (warmupErr != cudaSuccess)
    {
        printf("Warmup failed: %s\n",
            cudaGetErrorString(warmupErr));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(input);
        cudaFree(partial);
        cudaFree(result);
        return 1;
    }

    // //记录开始事件(单次测试)
    // cudaEventRecord(start);
    // //启动需要测量的kernel
    // largeReduction<<<NUM_BLOCKS,BLOCK_SIZE>>>(
    //     input,partial,N
    // );
    // //记录结束事件
    // cudaEventRecord(stop);
    // //等待结束事件完成
    // cudaEventSynchronize(stop);
    // //获取执行时间
    // float elapsedMs = 0.0f;
    // cudaEventElapsedTime(&elapsedMs,start,stop);
    // printf("kernel 1 time = %f ms\n",elapsedMs);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);


    //正式重复测试
    const int REPEAT = 100;
    //记录开始事件
    cudaEventRecord(start);
    //连续执行100次kernel
    for(int i = 0;i<REPEAT;i++)
    {
        largeReduction<<<NUM_BLOCKS,BLOCK_SIZE>>>(
         input,partial,N
    );
    }
    // 记录结束事件
    cudaEventRecord(stop);

    // 等待所有操作完成
    cudaError_t timingErr = cudaEventSynchronize(stop);

    if (timingErr != cudaSuccess)
    {
    printf("Benchmark failed: %s\n",
           cudaGetErrorString(timingErr));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
    cudaFree(partial);
    cudaFree(result);
    return 1;
    }

    // 获取总耗时
    float totalMs = 0.0f;

    cudaEventElapsedTime(&totalMs, start, stop);

    // 计算平均耗时
    float avgMs = totalMs / REPEAT;

    printf("Kernel 1 total time = %f ms\n", totalMs);
    printf("Kernel 1 average time = %f ms\n", avgMs);

    //===========================
    //kernel 2 性能测试
    //=============================

    //预热kernel 2
    for(int i = 0 ; i < WARMUP; i++)
    {
        finalReduction<<<1,BLOCK_SIZE>>>(
            partial,result,NUM_BLOCKS
        );
    }
    //等待预热结束
    cudaError_t warmupErr2 = cudaDeviceSynchronize();
    if(warmupErr2 != cudaSuccess)
    {
        printf("kernel 2 warmup failed!:%s\n",cudaGetErrorString(warmupErr2));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(input);
        cudaFree(result);
        cudaFree(partial);

        return 1;
    }

    //记录开始时间
    cudaEventRecord(start);
    //正式执行100次
    for(int i = 0;i < REPEAT; i++)
    {
        finalReduction<<<1,BLOCK_SIZE>>>(
            partial,result,NUM_BLOCKS
        );
    }
    //记录结束时间
    cudaEventRecord(stop);

    // 等待 GPU 完成
    cudaError_t timingErr2 = cudaEventSynchronize(stop);

    if (timingErr2 != cudaSuccess)
    {
        printf("Kernel 2 benchmark failed: %s\n",
            cudaGetErrorString(timingErr2));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(input);
        cudaFree(partial);
        cudaFree(result);
        return 1;
    }

    //计算执行时间
    float totalMs2 = 0.0f;

    cudaEventElapsedTime(&totalMs2,start,stop);
    float avgMs2 = totalMs2/REPEAT;

    printf("kernel 2 total time = %f ms\n",totalMs2);
    printf("kernel 2 average time = %f ms\n",avgMs2);

    // ===================================
    // 两阶段 Reduction 整体性能测试
    // ===================================

    // 预热
    for (int i = 0; i < WARMUP; i++)
    {
        largeReduction<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            input, partial, N
        );

        finalReduction<<<1, BLOCK_SIZE>>>(
            partial, result, NUM_BLOCKS
        );
    }

    // 等待预热完成
    cudaError_t warmupErr3 = cudaDeviceSynchronize();

    if (warmupErr3 != cudaSuccess)
    {
        printf("Two-stage warmup failed: %s\n",
            cudaGetErrorString(warmupErr3));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(input);
        cudaFree(partial);
        cudaFree(result);
        return 1;
    }

    // 记录开始事件
    cudaEventRecord(start);

    // 连续执行 100 次完整的两阶段 Reduction
    for (int i = 0; i < REPEAT; i++)
    {
        // 第一阶段
        largeReduction<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            input, partial, N
        );

        // 第二阶段
        finalReduction<<<1, BLOCK_SIZE>>>(
            partial, result, NUM_BLOCKS
        );
    }

    // 记录结束事件
    cudaEventRecord(stop);

    // 等待所有 Kernel 执行完成
    cudaError_t timingErr3 = cudaEventSynchronize(stop);

    if (timingErr3 != cudaSuccess)
    {
        printf("Two-stage benchmark failed: %s\n",
            cudaGetErrorString(timingErr3));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(input);
        cudaFree(partial);
        cudaFree(result);
        return 1;
    }

    // 计算总耗时
    float totalMs3 = 0.0f;

    cudaEventElapsedTime(&totalMs3, start, stop);

    // 计算每次两阶段 Reduction 的平均耗时
    float avgMs3 = totalMs3 / REPEAT;

    printf("Two-stage total time = %f ms\n", totalMs3);
    printf("Two-stage average time = %f ms\n", avgMs3);

    // 再次验证最终结果
    printf("Benchmark final sum = %d\n", result[0]);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 释放内存
    cudaFree(input);
    cudaFree(partial);
    cudaFree(result);
    

    return 0;
}


