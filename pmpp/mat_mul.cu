#include <vector>
#include <iostream>

// One result element per thread
__global__
void matMulKernel(float* A, float* B, float* C, int size) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if ((row < size) && (col < size)) {
        float res = 0;
        for (int k = 0; k < size; k++) {
            res += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = res;
    }
}

// One result row per thread
__global__
void matMulKernel2(float* A, float* B, float* C, int size) {
    // TODO
}

void matMul(float* A, float* B, float* C, int n) {
    float* A_d, * B_d, * C_d;
    int size = n * sizeof(float);
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    int gridSize = n;
    dim3 blockDim{ 256, 256, 1 };
    dim3 gripDim{ ceil(gridSize / 256.0), ceil(gridSize / 256.0), 1 };
    matMulKernel << < gridDim, blockDim >> > (A_d, B_d, C_d, n);

    // int gridSize = n;
    // dim3 blockDim{ 256, 1, 1 };
    // dim3 gripDim{ ceil(gridSize / 256.0), 1, 1 };
    // matMulKernel2 << < gridDim, blockDim >> > (A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    const auto N = 1000;
    std::vector<float> A, B;
    std::vector<float> C(A.size());
    matMul(A.data(), B.data(), C.data(), N);
}