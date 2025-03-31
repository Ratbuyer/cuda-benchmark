#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (32 * WARPS_PER_BLOCK)
#define BLOCKS_PER_GRID 16

#define BLOCK_SIZE 4

#include <cuda/barrier>

// Suppress warning about barrier in shared memory
#pragma nv_diag_suppress static_var_with_dynamic_init

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// kernel 1
__global__ void kernel_stride(int * data, int size, int work_per_block, int *results) {
	
	const int warp_id = (threadIdx.x / 32) + (blockIdx.x * WARPS_PER_BLOCK);
	const int block_id = blockIdx.x;
	const int tid = threadIdx.x % 32;
	const int thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
	
	const int shared_size = 32 * WARPS_PER_BLOCK * BLOCK_SIZE;
	
	__align__(128) __shared__ int shared[32 * WARPS_PER_BLOCK * BLOCK_SIZE];
	
	__shared__ barrier bar;
	
	if (threadIdx.x == 0) {
		init(&bar, blockDim.x);
	}
	
	uint64_t token;
	
	int sum = 0;
	
	for (int i = 0; i < work_per_block/shared_size; ++i) {
		if (tid == 0) {
			// call the loading api
			cde::cp_async_bulk_global_to_shared(shared, data + block_id * work_per_block + i * shared_size, sizeof(shared), bar);
			token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(shared));
		} else {
			token = bar.arrive();
		}
		
		bar.wait(std::move(token));
		
		for (int j = 0; j < BLOCK_SIZE; j++) {
			sum += shared[thread_id * BLOCK_SIZE + j];
		}
	}
	
	results[thread_id] = sum;
	
}

// kernel 2
__global__ void kernel_contiguous(int * data, int size, int work_per_warp, int *results) {
	
	const int warp_id = (threadIdx.x / 32) + (blockIdx.x * WARPS_PER_BLOCK);
	const int tid = threadIdx.x % 32;
	const int thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
	
	int4 buffer;
	int sum = 0;
	
	int stride = 32 * 4 * WARPS_PER_BLOCK * BLOCKS_PER_GRID * BLOCK_SIZE;
	int stride4 = stride / 4;
	int steps = size / stride;
	
	int4 *data4 = reinterpret_cast<int4 *>(data);
	
	for (int i = 0; i < steps; i++) {
		for (int j = 0; j < BLOCK_SIZE; j++) {
			buffer = data4[i * stride4 + warp_id * 32 * BLOCK_SIZE + tid * BLOCK_SIZE + j];
			sum += buffer.x + buffer.y + buffer.z + buffer.w;
		}
	}
	
	// printf("Warp %d, sum: %d\n", warp_id, sum);
	
	results[thread_id] = sum;
	
}