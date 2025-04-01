#include <stdio.h>
#include <cassert>
#include <random>

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
	
	constexpr int total_size = 1 << 28;
	
	int work_per_block = total_size / (BLOCKS_PER_GRID);
	
	assert(work_per_block % (32 * 4) == 0);
	
	printf("Data size in GB: %f\n", total_size * sizeof(int) / 1e9);
	printf("Work per block: %d\n", work_per_block);
	
	int *data = new int[total_size];
	
	for (int i = 0; i < total_size; i++) {
		data[i] = rand() % 10;
	}
	
	int *d_data, *d_results;
	cudaMalloc(&d_data, total_size * sizeof(int));
	
	cudaMemcpy(d_data, data, total_size * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc(&d_results, WARPS_PER_BLOCK * BLOCKS_PER_GRID * 32 * sizeof(int));
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	for (int i = 0; i < 1000; i++) {
		#if KERNEL == 1
		kernel_stride3<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_data, total_size, work_per_block, d_results);
		#elif KERNEL == 2
		kernel_contiguous<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_data, total_size, work_per_block, d_results);
		#endif
	}
	
	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	printf("Time: %f ms\n", milliseconds);
	
	cuda_check_error();
	
	int *results = new int[WARPS_PER_BLOCK * BLOCKS_PER_GRID * 32];
	
	cudaMemcpy(results, d_results, WARPS_PER_BLOCK * BLOCKS_PER_GRID * 32 * sizeof(int), cudaMemcpyDeviceToHost);
	
	int sum = 0;
	for (int i = 0; i < WARPS_PER_BLOCK * BLOCKS_PER_GRID * 32; i++) {
		sum += results[i];
	}
	
	int cpu_sum = 0;
	for (int i = 0; i < total_size; i++) {
		cpu_sum += data[i];
	}
	
	printf("Sum: %d\n", sum);
	printf("CPU Sum: %d\n", cpu_sum);
	
	return 0;
}