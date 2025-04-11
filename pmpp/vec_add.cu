#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <chrono>
#include "utils.hpp"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::microseconds;

void vecAddCPU(float* A, float* B, float* C, int n) {
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
    auto end = high_resolution_clock::now();
    std::cout << "Vec add (CPU) took " << duration_cast<microseconds>(end - start).count() << " microseconds" << std::endl;
}

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int n) {
    auto start = high_resolution_clock::now();
    float* A_d, * B_d, * C_d;
    int size = n * sizeof(float);
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    auto mem_copy_to_gpu_end = high_resolution_clock::now();
    std::cout << "Memory copy to GPU took " << duration_cast<microseconds>(mem_copy_to_gpu_end - start).count() << " microseconds" << std::endl;
    vecAddKernel << < ceil(n / 256.0), 256 >> > (A_d, B_d, C_d, n);
    auto kernel_call_end = high_resolution_clock::now();
    std::cout << "Kernel call took " << duration_cast<microseconds>(kernel_call_end - mem_copy_to_gpu_end).count() << " microseconds" << std::endl;
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    auto mem_copy_to_host_end = high_resolution_clock::now();
    std::cout << "Memory copy to host took " << duration_cast<microseconds>(mem_copy_to_host_end - kernel_call_end).count() << " microseconds" << std::endl;
    std::cout << "Vec add (GPU) took " << duration_cast<microseconds>(mem_copy_to_host_end - start).count() << " microseconds" << std::endl;
}

int main() {
    std::vector<float> A, B;
    readInput("in.txt", A);
    readInput("in.txt", B);
    
    std::vector<float> C(A.size());
    vecAdd(A.data(), B.data(), C.data(), A.size());
    printVectors(A, B, C);

    vecAddCPU(A.data(), B.data(), C.data(), A.size());
    std::cout << "\n";
}