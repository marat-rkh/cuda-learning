
#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

void readInput(const std::string& filename, std::vector<float>& data) {
    std::ifstream file {filename};
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for reading!" << std::endl;
        return;
    }
    std::string line;
    float num;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        while (iss >> num) {
            data.push_back(num);
        }
    }
    file.close();
}

void printVector(const std::vector<float>& vec, const std::string& name) {
    std::cout << name << ": ";
    for (size_t i = 0; i < 10 && i < vec.size(); i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << "\n";
}

void printVectors(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C) {
    std::cout << "Vectors added. The first 10 values:\n";
    printVector(A, "A");
    printVector(B, "B");
    printVector(C, "C");
    std::cout << "\n";
}