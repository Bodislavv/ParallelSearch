#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>
#include <fstream>
#include <string>

const std::string FILENAME_1B = "array_1B.bin";
const std::string FILENAME_500M = "array_500M.bin";

int main() {
    std::vector<int> elements;
    long long actual_total_elements = 0;
    std::string filename_to_process;

    int choice;
    std::cout << "Choose array to process:" << std::endl;
    std::cout << "1. Array (" << FILENAME_1B << ")" << std::endl;
    std::cout << "2. Array (" << FILENAME_500M << ")" << std::endl;
    std::cout << "Enter your choice (1 or 2): ";
    std::cin >> choice;

    if (choice == 1) {
        filename_to_process = FILENAME_1B;
    }
    else {
        filename_to_process = FILENAME_500M;
    }

    std::ifstream infile(filename_to_process, std::ios::binary | std::ios::in);

    infile.read(reinterpret_cast<char*>(&actual_total_elements), sizeof(actual_total_elements));
    elements.resize(actual_total_elements);
    if (actual_total_elements > 0) {
        infile.read(reinterpret_cast<char*>(elements.data()), actual_total_elements * sizeof(int));
    }
    infile.close();

    double search_start_time = omp_get_wtime();

    int global_min = INT_MAX;
    int global_max = INT_MIN;

#pragma omp parallel for reduction(min:global_min) reduction(max:global_max) schedule(static)
    for (long long i = 0; i < actual_total_elements; ++i) {
        if (elements[i] < global_min) {
            global_min = elements[i];
        }
        if (elements[i] > global_max) {
            global_max = elements[i];
        }
    }

    double search_end_time = omp_get_wtime();

    std::cout << "Global minimum: " << global_min << std::endl;
    std::cout << "Global maximum: " << global_max << std::endl;
    std::cout << "Time taken: " << (search_end_time - search_start_time) << " seconds" << std::endl;

    return 0;
}