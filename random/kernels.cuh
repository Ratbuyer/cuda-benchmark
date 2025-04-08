#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (32 * WARPS_PER_BLOCK)
#define BLOCKS_PER_GRID 8

#include <cstdint>

// kernel 1
__global__ void kernel_int4(int * data, int * d_slots, uint64_t slots_per_warp, uint64_t num_slots, int *results) {
	
	const int warp_id = (threadIdx.x / 32) + (blockIdx.x * WARPS_PER_BLOCK);
	const int tid = threadIdx.x % 32;
	const int thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
	
	int4 buffer;
	int sum = 0;
	
	// move the pointer to the start of the data for this warp, leaving gaps between warps
	int4 *data4 = reinterpret_cast<int4 *>(data);
	
	for (int i = 0; i < slots_per_warp; i++) {
		
		int slot_id = d_slots[i * WARPS_PER_BLOCK * BLOCKS_PER_GRID + warp_id];
		
		for (int j = 0; j < BLOCK_SIZE; j++) {
			buffer = data4[slot_id * 32 * BLOCK_SIZE + j * 32 + tid];
			sum += buffer.x + buffer.y + buffer.z + buffer.w;
		}
	}
	
	// printf("Warp %d, sum: %d\n", warp_id, sum);
	
	results[thread_id] = sum;
}

__global__ void kernel_int(int * data, int * d_slots, uint64_t slots_per_warp, uint64_t num_slots, int *results) {
	
	const int warp_id = (threadIdx.x / 32) + (blockIdx.x * WARPS_PER_BLOCK);
	const int tid = threadIdx.x % 32;
	const int thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
	
	int buffer;
	int sum = 0;
	
	for (int i = 0; i < slots_per_warp; i++) {
		
		int slot_id = d_slots[i * WARPS_PER_BLOCK * BLOCKS_PER_GRID + warp_id];
		
		for (int j = 0; j < BLOCK_SIZE; j++) {
			buffer = data[slot_id * 32 * BLOCK_SIZE + j * 32 + tid];
			sum += buffer;
		}
	}
	
	results[thread_id] = sum;
}