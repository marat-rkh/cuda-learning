#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <chrono>

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

constexpr int N = 1000000;

void generateRandomInput() {
    std::ofstream file("in.txt");
    if (!file) {
        std::cerr << "Error: Cannot open file for writing!" << std::endl;
        return;
    }
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister PRNG
    std::uniform_real_distribution<float> dist(1, 1000);
    for (size_t i = 0; i < N; i++) {
        file << dist(gen) << " ";
    }
    file << "\n";
    for (size_t i = 0; i < N; i++) {
        file << dist(gen) << " ";
    }
    file.close();
}

void readInput(std::vector<float>& A, std::vector<float>& B) {
    std::ifstream file("in.txt");
    if (!file) {
        std::cerr << "Error: Cannot open file for reading!" << std::endl;
        return;
    }
    std::string line;
    float num;
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        while (iss >> num) {
            A.push_back(num);
        }
    }
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        while (iss >> num) {
            B.push_back(num);
        }
    }
    file.close();
}

int main() {
    generateRandomInput();

    std::vector<float> A, B;
    readInput(A, B);
    
    std::vector<float> C(A.size());
    vecAdd(A.data(), B.data(), C.data(), N);

    std::cout << "Vectors added. The first 10 values:\n";
    std::cout << "A: ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << "\n";
    std::cout << "B: ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << B[i] << " ";
    }
    std::cout << "\n";
    std::cout << "C: ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << "\n\n";

    vecAddCPU(A.data(), B.data(), C.data(), N);
    std::cout << "\n";
}