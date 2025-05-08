#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <mpi.h>
#include <algorithm>
#include <random>

std::mt19937 generator;

void initialize_random_engine_uniquely(unsigned int base_seed, int rank) {
    generator.seed(base_seed + rank);
}

int get_random(int min_val, int max_val) {
    std::uniform_int_distribution<int> distribution(min_val, max_val);
    return distribution(generator);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const long long TOTAL_ELEMENTS = 1000000000LL;

    long long elements_per_proc = TOTAL_ELEMENTS / world_size;
    long long remainder = TOTAL_ELEMENTS % world_size;
    long long local_n = elements_per_proc + (world_rank < remainder ? 1 : 0);

    std::vector<int> local_elements(local_n);

    initialize_random_engine_uniquely(static_cast<unsigned int>(time(NULL)), world_rank);

    for (long long i = 0; i < local_n; ++i) {
        local_elements[i] = get_random(0, 1000000000);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    int local_min = INT_MAX;
    int local_max = INT_MIN;

    if (local_n > 0) {
        local_min = local_elements[0];
        local_max = local_elements[0];
        for (long long i = 1; i < local_n; ++i) {
            if (local_elements[i] < local_min) {
                local_min = local_elements[i];
            }
            if (local_elements[i] > local_max) {
                local_max = local_elements[i];
            }
        }
    }

    int global_min;
    int global_max;

    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (world_rank == 0) {
        std::cout << "Global minimum: " << global_min << std::endl;
        std::cout << "Global maximum: " << global_max << std::endl;
        std::cout << "Time taken for search (including reductions): " << (end_time - start_time) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
