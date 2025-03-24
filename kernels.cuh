#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (32 * WARPS_PER_BLOCK)
#define BLOCKS_PER_GRID 8


__global__ void kernel_stride(int * data, int size, int work_per_warp, int *results) {
	
	const int warp_id = (threadIdx.x / 32) + (blockIdx.x * WARPS_PER_BLOCK);
	const int tid = threadIdx.x % 32;
	const int thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
	
	int4 buffer;
	int sum = 0;
	
	int4 *data4 = reinterpret_cast<int4 *>(data + warp_id * work_per_warp);
	
	for (int i = 0; i < work_per_warp / (32 * 4); i++) {
		buffer = data4[i * 32 + tid];
		sum += buffer.x + buffer.y + buffer.z + buffer.w;
	}
	
	// printf("Warp %d, sum: %d\n", warp_id, sum);
	
	results[thread_id] = sum;
	
}

__global__ void kernel_contiguous(int * data, int size, int work_per_warp, int *results) {
	
	const int warp_id = (threadIdx.x / 32) + (blockIdx.x * WARPS_PER_BLOCK);
	const int tid = threadIdx.x % 32;
	const int thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
	
	int4 buffer;
	int sum = 0;
	
	int stride = 32 * 4 * WARPS_PER_BLOCK * BLOCKS_PER_GRID;
	int stride4 = 32 * WARPS_PER_BLOCK * BLOCKS_PER_GRID;
	int steps = size / stride;
	
	int4 *data4 = reinterpret_cast<int4 *>(data);
	
	for (int i = 0; i < steps; i++) {
		buffer = data4[i * stride4 + warp_id * 32 + tid];
		sum += buffer.x + buffer.y + buffer.z + buffer.w;
	}
	
	// printf("Warp %d, sum: %d\n", warp_id, sum);
	
	results[thread_id] = sum;
	
}