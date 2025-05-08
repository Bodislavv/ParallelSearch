#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <algorithm>
#include <random>
#include <omp.h>

int main() {
    const long long TOTAL_ELEMENTS = 1000000000LL;
    std::vector<int> elements(TOTAL_ELEMENTS);

    double gen_start_time = omp_get_wtime();

#pragma omp parallel
    {
        unsigned int seed = static_cast<unsigned int>(time(NULL)) ^ (omp_get_thread_num() << 16);
        std::mt19937 generator(seed);
        std::uniform_int_distribution<int> distribution(0, 1000000000);

#pragma omp for schedule(static)
        for (long long i = 0; i < TOTAL_ELEMENTS; ++i) {
            elements[i] = distribution(generator);
        }
    }
    double gen_end_time = omp_get_wtime();

    double search_start_time = omp_get_wtime();

    int global_min = INT_MAX;
    int global_max = INT_MIN;

#pragma omp parallel for reduction(min:global_min) reduction(max:global_max) schedule(static)
    for (long long i = 0; i < TOTAL_ELEMENTS; ++i) {
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
    std::cout << "Time taken for search: " << (search_end_time - search_start_time) << " seconds" << std::endl;

    return 0;
}
