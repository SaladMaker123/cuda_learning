#include <stdio.h>

//定义全局托管变量
__managed__ int counter =100;
//gpu kernel
__global__ void modifyCounter()
{
    counter = counter + 50;

    printf("GPU: counter = %d\n", counter);
}

int main()
{
    // 第一步：CPU 读取初始值
    printf("CPU: Initial counter = %d\n", counter);

    // 第二步：CPU 修改托管变量
    counter = 200;

    printf("CPU: After modification = %d\n", counter);

    // 第三步：启动 GPU Kernel
    modifyCounter<<<1, 1>>>();

    // 第四步：检查 Kernel 启动错误
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    // 第五步：等待 GPU 执行完成
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("Kernel execution failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    // 第六步：CPU 读取 GPU 修改后的值
    printf("CPU: Final counter = %d\n", counter);

    return 0;
}