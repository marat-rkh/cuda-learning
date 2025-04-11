#include <fstream>
#include <iostream>
#include <random>

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: gen_input <name> <size> <size>" << std::endl;
        return 1;
    }
    std::string filename {argv[1]};
    int N = std::stoi(argv[2]);
    int M = std::stoi(argv[3]);
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open file for writing!" << std::endl;
        return 1;
    }
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister PRNG
    std::uniform_real_distribution<float> dist(1, 1000);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            file << dist(gen) << " ";
        }
        file << "\n";
    }
    file.close();
    return 0;
}