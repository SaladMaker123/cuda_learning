#include <cuda_runtime.h>
#include <stdio.h>

__global__ void partialSumkernel(
    const int* input,
    int* partial,
    int N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int inputIndex = idx *2;
    if(inputIndex+1<N)
    {
        partial[idx] = input[inputIndex] + input[inputIndex+1];
    }
}


int main()
{
    const int N =8;
    const int partialN = N/2;
    const int secondN = partialN/2;


    int* input = nullptr;
    int* partial = nullptr;
    int* second = nullptr;
    int* result = nullptr;

    //分配统一内存
    cudaMallocManaged(&input,N*sizeof(int));
    cudaMallocManaged(&partial,partialN*sizeof(int));
    cudaMallocManaged(&second,secondN*sizeof(int));
    cudaMallocManaged(&result, sizeof(int));

    //初始化输入数组
    for(int i = 0;i<N;i++)
    {
        input[i] = i + 1;
    }

    //启动kernel
    partialSumkernel<<<1,4>>>(input,partial,N);
    //等待GPU完成
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        printf("kernel execution failed!:%s",cudaGetErrorString(err));
        cudaFree(input);
        cudaFree(partial);
        return 1;
    }
    //输出部分和

    for(int i = 0;i<partialN;i++)
    {
        printf("partial[%d]=%d\n",i,partial[i]);
    }

    //第二轮规约
    partialSumkernel<<<1,2>>>(partial,second,partialN);
    err = cudaDeviceSynchronize();
    if(err !=cudaSuccess)
    {
        printf("second execution failed:%s",cudaGetErrorString(err));
        cudaFree(input);
        cudaFree(partial);
        return 1;
    }
    //输出第二轮结果
    for(int i = 0;i<secondN;i++)
    {
        printf("second[%d]=%d\n",i,second[i]);
    }

    // 第三轮归约
    partialSumkernel<<<1, 1>>>(second, result, secondN);

    // 等待 GPU 完成
    cudaError_t err3 = cudaDeviceSynchronize();

    if (err3 != cudaSuccess)
    {
        printf("Third kernel failed: %s\n", cudaGetErrorString(err3));
        cudaFree(input);
        cudaFree(partial);
        cudaFree(second);
        cudaFree(result);
        return 1;
    }

    // 输出最终结果
    printf("Final sum = %d\n", result[0]);


    cudaFree(input);
    cudaFree(partial);
    cudaFree(second);
    cudaFree(result);
    return 0;


}