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

__device__ __forceinline__ void load_recursive(int4 * data, int * sum, int start, int end) {
	
	const int warp_id = (threadIdx.x / 32) + (blockIdx.x * WARPS_PER_BLOCK);
	const int thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
	const int tid = threadIdx.x % 32;
	
	int4 buffer;
	
	if (start == end) {
		
		// if (true) {
		// 	printf("start: %d, end: %d\n", start, end);
		// }
		
		for (int i = 0; i < BLOCK_SIZE; i++) {
			buffer = data[start * 32 * BLOCK_SIZE + i * 32 + tid];
			*sum += buffer.x + buffer.y + buffer.z + buffer.w;
		}
		
		return;
	}
	
	int mid = (end + start) / 2;
	
	// if (tid == 0) {
	// 	printf("start: %d, mid: %d, end: %d\n", start, mid, end);
	// }
	
	// load mid block
	for (int i = 0; i < BLOCK_SIZE; i++) {
		buffer = data[mid * 32 * BLOCK_SIZE + i * 32 + tid];
		*sum += buffer.x + buffer.y + buffer.z + buffer.w;
	}
	
	if (end != mid) load_recursive(data, sum, mid + 1, end);
	
	if (start != mid) load_recursive(data, sum, start, mid - 1);

}

// kernel 3
__global__ void kernel_gap(int * data, int size, int work_per_warp, int *results) {
	
	const int warp_id = (threadIdx.x / 32) + (blockIdx.x * WARPS_PER_BLOCK);
	// const int tid = threadIdx.x % 32;
	const int thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
	
	int sum = 0;
	
	// move the pointer to the start of the data for this warp, leaving gaps between warps
	int4 *data4 = reinterpret_cast<int4 *>(data + warp_id * work_per_warp);
	
	int end = work_per_warp / (32 * 4 * BLOCK_SIZE) - 1;
	
	load_recursive(data4, &sum, 0, end);
	
	// printf("Warp %d, sum: %d\n", warp_id, sum);
	
	results[thread_id] = sum;
	
}