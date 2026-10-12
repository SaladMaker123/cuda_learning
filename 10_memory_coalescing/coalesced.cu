#include <cuda_runtime.h>
#include <stdio.h>

__global__ void coalescedKernel(
    const float* input,
    float* output,
    int N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        output[idx] = input[idx] * 2.0f;
    }
}


__global__ void stridedKernel(
    const float* input,
    float* output,
    int N,
    int stride
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        int inputIndex = idx * stride;
        output[idx] = input[inputIndex] * 2.0f;
    }
}
int main()
{
    const int N = 250;
    const int stride = 32;
    float* input = nullptr;
    float* output = nullptr;
    size_t bytes = N *sizeof(float);
    size_t inputBytes = N * stride * sizeof(float);
    size_t outputBytes = N * sizeof(float);

    //分配unified Memory
    cudaError_t err = cudaMallocManaged(&input,inputBytes);
    if(err != cudaSuccess)
    {
        printf("Input allocation failed!:%s\n",cudaGetErrorString(err));
        return 1;
    }
    err = cudaMallocManaged(&output,outputBytes);
    if(err != cudaSuccess)
    {
        printf("Output allocation failed!\n");
        cudaFree(input);
        return 1;
    }

    //cpu初始化数组
    for(int i = 0;i<N*stride;i++)
    {
        input[i] = i + 1.0f;
    }

    //启动kernel
    // coalescedKernel<<<2,128>>>(input,output,N);
    stridedKernel<<<2,128>>>(input,output,N,stride);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(input);
        cudaFree(output);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        printf("kernel execution failed!%s\n",cudaGetErrorString(err));
        cudaFree(input);
        cudaFree(output);
        return 1;
    }

    //验证所有输出元素
    bool correct = true;
    for(int i = 0;i<N;i++)
    {
        if(output[i] != input[i*stride]*2.0f)
        {
            correct = false;
            break;
        }
    }

    printf("input[224] = %.1f\n",input[224]);
    printf("output[7]=%.1f\n",output[7]);
    printf("result:%s\n",correct?"PASS":"FAIL");

    cudaFree(input);
    cudaFree(output);
    return correct ? 0:1;
}