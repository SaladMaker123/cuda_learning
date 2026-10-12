#include <cuda_runtime.h>
#include <stdio.h>

//kernel A:连续访问
__global__ void coalescedKernel(
    const float* input,
    float* output,
    int N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N)
    {
        output[idx] = input[idx] *2.0f;
    }
}

//kernel B:跨步访问
__global__ void stridedKernel(
    const float* input,
    float* output,
    int N,
    int stride
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N )
    {
        int row = idx / stride;
        int col = idx % stride;
        int inputIndex = col*(N/stride) + row;
        
        output[idx] = input[inputIndex]*2.0f;
    }
}

//计时函数1
float measureCoalesced(
    const float* d_input,
    float* d_output,
    int N,
    int blocks,
    int threadsPerBlock
)
{
    const int warmup = 5;
    const int repeat = 100;
    //GPU预热
    for(int i = 0;i<warmup;i++)
    {
        coalescedKernel<<<blocks,threadsPerBlock>>>(
            d_input,d_output,N
        );
    };
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //正式重复测量
    for(int i = 0;i<repeat;i++){
        coalescedKernel<<<blocks,threadsPerBlock>>>(
        d_input,
        d_output,
        N);
    };
   
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float totalTime = 0.0f;
    cudaEventElapsedTime(&totalTime,start,stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return totalTime/repeat;
}

//计时函数2
float measureStrided(
    const float* d_input,
    float* d_output,
    int N,
    int stride,
    int blocks,
    int threadsPerBlock
)
{
    const int warmup = 5;
    const int repeat = 100;
    //GPU预热
    for(int i =0;i<warmup;i++)
    {
        stridedKernel<<<blocks,threadsPerBlock>>>(d_input,d_output,N,stride);
    };
    //正式测量
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    cudaEventRecord(start);
    for(int i =0;i<repeat;i++)
    {
        stridedKernel<<<blocks,threadsPerBlock>>>(
        d_input,
        d_output,
        N,
        stride
    );
    };
    
    
    //记录结束时间
    cudaEventRecord(stop);
    //等待EVENT完成
    cudaEventSynchronize(stop);
    float totalTime = 0.0f;
    cudaEventElapsedTime(
        &totalTime,
        start,stop
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return totalTime/repeat;
}




int main()
{
    const int N = 8 * 1024 * 1024;
    const int stride = 32;
    size_t bytes = N * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaError_t err;

    //在gpu global memory中分配数组
    err = cudaMalloc(&d_input,bytes);
    if(err != cudaSuccess)
    {
        printf("input allocation failed!:%s\n",cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_output,bytes);
    if(err != cudaSuccess)
    {
        printf("output allocation failed!:%s\n",cudaGetErrorString(err));
        cudaFree(d_input);
        return 1;
    }

    printf("GPU memory allocation successful!\n");

    //在CPU内存中分配输入数组
    float* h_input = new float[N];
    float* h_output = new float[N];
    //初始化CPU数组
    for(int i = 0;i<N;i++)
    {
        h_input[i] = i+1.0f;
    }

    //cpu数组复制到GPU
    err = cudaMemcpy(
        d_input,
        h_input,
        bytes,
        cudaMemcpyHostToDevice
    );
        if (err != cudaSuccess)
    {
        printf("Host to Device copy failed: %s\n",
            cudaGetErrorString(err));

        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    printf("Host to Device copy successful!\n");

    int threadsPerBlock = 256;
    int blocks = (N+threadsPerBlock-1)/threadsPerBlock;
    coalescedKernel<<<blocks,threadsPerBlock>>>(
        d_input,
        d_output,
        N
    );
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n",
            cudaGetErrorString(err));

        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }   

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("Kernel execution failed: %s\n",
            cudaGetErrorString(err));

        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    err = cudaMemcpy(
    h_output,
    d_output,
    bytes,
    cudaMemcpyDeviceToHost
    );

    if (err != cudaSuccess)
    {
        printf("Device to Host copy failed: %s\n",
            cudaGetErrorString(err));

        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }


    bool correct = true;

    for (int i = 0; i < N; i++)
    {
        if (h_output[i] != h_input[i] * 2.0f)
        {
            correct = false;
            break;
        }
    }

    printf("Coalesced Kernel: %s\n",
        correct ? "PASS" : "FAIL");

    printf("h_output[100] = %.1f\n", h_output[100]);

    //启动跨步访存kernel
    stridedKernel<<<blocks,threadsPerBlock>>>(
        d_input,
        d_output,
        N,
        stride
    );
    // 检查 Kernel 启动错误
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Strided Kernel launch failed: %s\n",
            cudaGetErrorString(err));

        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    //等待GPU执行完成
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        printf("kernel execution failed:%s\n",cudaGetErrorString(err));
        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    err = cudaMemcpy(
        h_output,
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    );
    if (err != cudaSuccess)
    {
        printf("Strided result copy failed: %s\n",
            cudaGetErrorString(err));

        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    bool stridedCorrect = true;

    for (int i = 0; i < N; i++)
    {
        int row = i / stride;
        int col = i % stride;

        int inputIndex = col * (N / stride) + row;

        if (h_output[i] != h_input[inputIndex] * 2.0f)
        {
            stridedCorrect = false;
            break;
        }
    }

    printf("Strided Kernel: %s\n",
        stridedCorrect ? "PASS" : "FAIL");

    printf("Strided h_output[35] = %.1f\n", h_output[35]);

    float timeCoalesced = measureCoalesced(
        d_input,
        d_output,
        N,
        blocks,
        threadsPerBlock
    );
    printf("Coalesced Kernel time: %f ms\n", timeCoalesced);


    float timeStrided = measureStrided(
    d_input,
    d_output,
    N,
    stride,
    blocks,
    threadsPerBlock
    );

    printf("Strided Kernel time: %f ms\n", timeStrided);


    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return (correct && stridedCorrect) ? 0 : 1;
}