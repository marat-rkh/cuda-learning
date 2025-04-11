#include <vector>
#include <iostream>
#include "utils.hpp"

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
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        for (int j = 0; j < size; j++) {
            float res = 0;
            for (int k = 0; k < size; k++) {
                res += A[idx * size + k] * B[k * size + j];
            }
            C[idx * size + j] = res;
        }
    }
}

// One result column per thread
__global__
void matMulKernel3(float* A, float* B, float* C, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        for (int j = 0; j < size; j++) {
            float res = 0;
            for (int k = 0; k < size; k++) {
                res += A[j * size + k] * B[idx * size + k];
            }
            C[j * size + idx] = res;
        }
    }
}

enum class Mode {
    ROW,
    COL,
    ELEM
};

void matMul(float* A, float* B, float* C, int n, Mode mode) {
    float* A_d, * B_d, * C_d;
    int size = n * sizeof(float);
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    if (mode == Mode::ELEM) {
        dim3 blockDim{ 256, 256, 1 };
        dim3 gripDim{ static_cast<unsigned int>(ceil(n / 256.0)), static_cast<unsigned int>(ceil(n / 256.0)), 1 };
        matMulKernel << < gridDim, blockDim >> > (A_d, B_d, C_d, n);
    } else if (mode == Mode::ROW) {
        dim3 blockDim{ 256, 1, 1 };
        dim3 gripDim{ static_cast<unsigned int>(ceil(n / 256.0)), 1, 1 };
        matMulKernel2 << < gridDim, blockDim >> > (A_d, B_d, C_d, n);
    } else if (mode == Mode::COL) {
        dim3 blockDim{ 256, 1, 1 };
        dim3 gripDim{ static_cast<unsigned int>(ceil(n / 256.0)), 1, 1 };
        matMulKernel3 << < gridDim, blockDim >> > (A_d, B_d, C_d, n);
    }

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    std::vector<float> A, B;
    readInput("A.txt", A);
    readInput("B.txt", B);
    std::vector<float> C(A.size());

    matMul(A.data(), B.data(), C.data(), A.size(), Mode::ELEM);
    printVectors(A, B, C);
    matMul(A.data(), B.data(), C.data(), A.size(), Mode::ROW);
    printVectors(A, B, C);
    matMul(A.data(), B.data(), C.data(), A.size(), Mode::COL);
    printVectors(A, B, C);
    return 0;
}