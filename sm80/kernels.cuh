#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (32 * WARPS_PER_BLOCK)
#define BLOCKS_PER_GRID 128

// kernel 1
__global__ void kernel_stride(int * data, int size, int work_per_warp, int *results) {
	
	const int warp_id = (threadIdx.x / 32) + (blockIdx.x * WARPS_PER_BLOCK);
	const int tid = threadIdx.x % 32;
	const int thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
	
	int4 buffer;
	int sum = 0;
	
	// move the pointer to the start of the data for this warp, leaving gaps between warps
	int4 *data4 = reinterpret_cast<int4 *>(data + warp_id * work_per_warp);
	
	for (int i = 0; i < work_per_warp / (32 * 4 * BLOCK_SIZE); i++) {
		for (int j = 0; j < BLOCK_SIZE; j++) {
			buffer = data4[i * 32 * BLOCK_SIZE + tid * BLOCK_SIZE + j];
			sum += buffer.x + buffer.y + buffer.z + buffer.w;
		}
	}
	
	// printf("Warp %d, sum: %d\n", warp_id, sum);
	
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