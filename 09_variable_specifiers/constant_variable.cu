#include<stdio.h>
//定义gpu常量内存变量
__constant__ float gpuScale =2.0f;

//gpu kernel
__global__ void scaleKernel(float* A,float*B,int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<N)
    {
        B[i] = A[i] *gpuScale;
    }
}

int main()
{
    int N =5;
    size_t size = N*sizeof(float);

    float* A = nullptr;
    float* B = nullptr;

    cudaError_t err;
    err = cudaMallocManaged(&A,size);

    if(err != cudaSuccess)
    {
        printf("Allocate A failed!:%s",cudaGetErrorString(err));
        return 1;
    }

    err = cudaMallocManaged(&B, size);

    if (err != cudaSuccess)
    {
        printf("Allocate B failed: %s\n",
               cudaGetErrorString(err));

        cudaFree(A);
        return 1;
    }

    //初始化数组
    for(int i = 0;i<N;i++)
    {
        A[i] = i+1;
        B[i] = 0;
    }

    //启动 kernel
    int threadsPerBlock = 4;
    int blocks = (N+threadsPerBlock-1)/threadsPerBlock;
    scaleKernel<<<blocks,threadsPerBlock>>>(A,B,N);

    //检查kernel启动错误
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        printf("kernel launch failed!:%s",cudaGetErrorString(err));
        cudaFree(A);
        cudaFree(B);
        return 1;
    }

    // 等待GPU计算完成
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        printf("kernel execution failed!:%s",cudaGetErrorString(err));
        cudaFree(A);
        cudaFree(B);
        return 1;
    }

    //输出计算结果
    for(int i =0;i<N;i++)
    {
        printf("A[%d] = %.1f,B[%d] = %.1f\n",i,A[i],i,B[i]);
    }

    // CPU 准备一个新的常量值
    float hostScale = 3.0f;

    //  将 CPU 数据复制到 GPU 常量内存
    err = cudaMemcpyToSymbol(
    gpuScale,
    &hostScale,
    sizeof(float)
    );

    if (err != cudaSuccess)
    {
    printf("Copy to constant memory failed: %s\n",
           cudaGetErrorString(err));

    cudaFree(A);
    cudaFree(B);
    return 1;
    }

    
    // 第二次启动 Kernel
    scaleKernel<<<blocks, threadsPerBlock>>>(A, B, N);
    // 检查第二次 Kernel 启动
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
    printf("Second Kernel launch failed: %s\n",
           cudaGetErrorString(err));

    cudaFree(A);
    cudaFree(B);
    return 1;
    }

    // 等待第二次 GPU 计算完成
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
    printf("Second Kernel execution failed: %s\n",
           cudaGetErrorString(err));

    cudaFree(A);
    cudaFree(B);
    return 1;
    }
    printf("\nAfter updating gpuScale to 3.0:\n");

    for (int i = 0; i < N; i++)
    {
        printf("A[%d] = %.1f, B[%d] = %.1f\n",
            i, A[i], i, B[i]);
    }
    cudaFree(A);
    cudaFree(B);

    return 0;
}
