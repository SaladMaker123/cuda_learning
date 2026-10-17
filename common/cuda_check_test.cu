#include <iostream>
#include "cuda_check.h"
int main()
{
    float* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, 1024 * sizeof(float)));
    std::cout << "cudaMalloc succeeded!" << std::endl;
    CUDA_CHECK(cudaFree(d_A));
    std::cout << "cudaFree succeeded!" << std::endl;

    return 0;
}

