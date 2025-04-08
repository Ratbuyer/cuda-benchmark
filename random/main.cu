#include <stdio.h>
#include <cassert>
#include <random>
#include <algorithm>
#include <iostream>

#include "kernels.cuh"

void cuda_check_error()
{
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

int main() {
	#if KERNEL == 1
	constexpr int slot_size = 32 * 4 * BLOCK_SIZE;
	#elif KERNEL == 2
	constexpr int slot_size = 32 * BLOCK_SIZE;
	#endif
	
	constexpr uint64_t total_size = (1ULL << DATA_SIZE);
	
	uint64_t num_slots = total_size / slot_size;
	uint64_t slots_per_warp = num_slots / (WARPS_PER_BLOCK * BLOCKS_PER_GRID);
	
	std::cout << "Data size in GB: " << total_size * sizeof(int) / 1e9 << std::endl;
	std::cout << "num slots: " << num_slots << std::endl;
	std::cout << "slots per warp: " << slots_per_warp << std::endl;
	
	// printf("Data size in GB: %f\n", total_size * sizeof(int) / 1e9);
	// printf("num slots: %ll\n", num_slots);
	// printf("slots per warp: %ll\n", slots_per_warp);
	
	assert(total_size % slot_size == 0);
	assert(num_slots % (WARPS_PER_BLOCK * BLOCKS_PER_GRID) == 0);
	
	int *slots = new int[num_slots];
	
	for (uint64_t i = 0; i < num_slots; i++) {
		slots[i] = i;
	}
	
	int *data = new int[total_size];
	
	for (uint64_t i = 0; i < total_size; i++) {
		data[i] = rand() % 10;
	}
	
	// permutate the num_slots
	std::random_shuffle(slots, slots + num_slots);
	
	int *d_data, *d_results;
	cudaMalloc(&d_data, total_size * sizeof(int));
	cudaMemcpy(d_data, data, total_size * sizeof(int), cudaMemcpyHostToDevice);
	
	int *d_slots;
	cudaMalloc(&d_slots, num_slots * sizeof(int));
	cudaMemcpy(d_slots, slots, num_slots * sizeof(int), cudaMemcpyHostToDevice);	
	
	cudaMalloc(&d_results, WARPS_PER_BLOCK * BLOCKS_PER_GRID * 32 * sizeof(int));
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	for (int i = 0; i < 1000; i++) {
		#if KERNEL == 1
		kernel_int4<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_data, d_slots, slots_per_warp, num_slots, d_results);
		#elif KERNEL == 2
		kernel_int<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_data, d_slots, slots_per_warp, num_slots, d_results);
		#endif
	}
	
	cuda_check_error();
	
	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	printf("Time: %f ms\n", milliseconds);
	
	int *results = new int[WARPS_PER_BLOCK * BLOCKS_PER_GRID * 32];
	
	cudaMemcpy(results, d_results, WARPS_PER_BLOCK * BLOCKS_PER_GRID * 32 * sizeof(int), cudaMemcpyDeviceToHost);
	
	int sum = 0;
	for (int i = 0; i < WARPS_PER_BLOCK * BLOCKS_PER_GRID * 32; i++) {
		sum += results[i];
	}
	
	int cpu_sum = 0;
	for (uint64_t i = 0; i < total_size; i++) {
		cpu_sum += data[i];
	}
	
	printf("Sum: %d\n", sum);
	printf("CPU Sum: %d\n", cpu_sum);
	
	assert(sum == cpu_sum);
	
	return 0;
}