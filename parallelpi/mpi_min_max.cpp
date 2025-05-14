#include <iostream>
#include <vector>
#include <string>
#include <climits>
#include <mpi.h>
#include <fstream>

const long long TOTAL_ELEMENTS_1B_EXPECTED = 1000000000LL;
const long long TOTAL_ELEMENTS_500M_EXPECTED = 500000000LL;
const std::string FILENAME_1B = "array_1B.bin";
const std::string FILENAME_500M = "array_500M.bin";

void load_array_from_file_no_checks(const std::string& filename, std::vector<int>& elements, long long& total_elements_from_file_header) {
    std::ifstream infile(filename, std::ios::binary | std::ios::in);
    infile.read(reinterpret_cast<char*>(&total_elements_from_file_header), sizeof(total_elements_from_file_header));
    elements.resize(total_elements_from_file_header);
    infile.read(reinterpret_cast<char*>(elements.data()), total_elements_from_file_header * sizeof(int));
    infile.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int choice = 0;
    std::string filename_to_process;
    long long actual_total_elements_from_file = 0;
    if (world_rank == 0) {
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
    }
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (choice == 1) {
        filename_to_process = FILENAME_1B;
    }
    else {
        filename_to_process = FILENAME_500M;
    }
    std::vector<int> global_elements_on_root;
    if (world_rank == 0) {
        load_array_from_file_no_checks(filename_to_process, global_elements_on_root, actual_total_elements_from_file);
    }
    MPI_Bcast(&actual_total_elements_from_file, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    long long total_elements_to_process = actual_total_elements_from_file;
    long long elements_per_proc = total_elements_to_process / world_size;
    long long remainder = total_elements_to_process % world_size;
    long long local_n = elements_per_proc + (world_rank < remainder ? 1 : 0);
    std::vector<int> local_elements(local_n);
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);
    if (world_rank == 0) {
        long long current_displ = 0;
        for (int i = 0; i < world_size; ++i) {
            sendcounts[i] = static_cast<int>(elements_per_proc + (i < remainder ? 1 : 0));
            displs[i] = static_cast<int>(current_displ);
            current_displ += sendcounts[i];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    MPI_Scatterv(
        world_rank == 0 ? global_elements_on_root.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_INT,
        local_elements.data(),
        static_cast<int>(local_n),
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );
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
    else {
        local_min = INT_MAX;
        local_max = INT_MIN;
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
        std::cout << "Time taken: " << (end_time - start_time) << " seconds" << std::endl;
    }
    MPI_Finalize();
    return 0;
}