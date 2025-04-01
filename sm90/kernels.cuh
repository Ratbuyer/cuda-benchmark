#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (32 * WARPS_PER_BLOCK)
#define BLOCKS_PER_GRID 8

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
		cde::fence_proxy_async_shared_cta();
	}
	
	__syncthreads();
	
	uint64_t token;
	
	int sum = 0;
	
	for (int i = 0; i < work_per_block/shared_size; ++i) {
		if (threadIdx.x == 0) {
			// call the loading api
			cde::cp_async_bulk_global_to_shared(shared, data + i * shared_size * BLOCKS_PER_GRID + block_id * shared_size, sizeof(shared), bar);
			token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(shared));
		} else {
			token = bar.arrive();
		}
		
		bar.wait(std::move(token));
		
		__syncthreads();
		
		for (int j = 0; j < BLOCK_SIZE; j++) {
			sum += shared[threadIdx.x * BLOCK_SIZE + j];
		}
		
		__syncthreads();
	}
	
	results[thread_id] = sum;
	
}

// kernel 2
__global__ void kernel_contiguous(int * data, int size, int work_per_block, int *results) {
	
	const int warp_id = (threadIdx.x / 32) + (blockIdx.x * WARPS_PER_BLOCK);
	const int block_id = blockIdx.x;
	const int tid = threadIdx.x % 32;
	const int thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
	
	const int shared_size = 32 * WARPS_PER_BLOCK * BLOCK_SIZE;
	
	__align__(128) __shared__ int shared[32 * WARPS_PER_BLOCK * BLOCK_SIZE];
	
	__shared__ barrier bar;
	
	if (threadIdx.x == 0) {
		init(&bar, blockDim.x);
		cde::fence_proxy_async_shared_cta();
	}
	
	__syncthreads();
	
	uint64_t token;
	
	int sum = 0;
	
	for (int i = 0; i < work_per_block/shared_size; ++i) {
		if (threadIdx.x == 0) {
			// call the loading api
			cde::cp_async_bulk_global_to_shared(shared, data + i * shared_size * BLOCKS_PER_GRID + block_id * shared_size, sizeof(shared), bar);
			token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(shared));
		} else {
			token = bar.arrive();
		}
		
		bar.wait(std::move(token));
		
		__syncthreads();
		
		for (int j = 0; j < BLOCK_SIZE; j++) {
			sum += shared[threadIdx.x * BLOCK_SIZE + j];
		}
		
		__syncthreads();
	}
	
	results[thread_id] = sum;
	
}


static constexpr size_t buf_len = 1024;

__global__ void test(int * data, int *results) {
	
	// Shared memory buffers for x and y. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                // a)
    cde::fence_proxy_async_shared_cta();   // b)
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_global_to_shared(smem_data, data, sizeof(smem_data), bar);
    // 3a. Arrive on the barrier and tell how many bytes are expected to come in (the transaction count)
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_data));
  } else {
    // 3b. Rest of threads just arrive
    token = bar.arrive();
  }
  
  results[0] = smem_data[0];
	
}

__global__ void add_one_kernel(int* data, size_t offset)
{
  // Shared memory buffers for x and y. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                // a)
    cde::fence_proxy_async_shared_cta();   // b)
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_global_to_shared(smem_data, data + offset, sizeof(smem_data), bar);
    // 3a. Arrive on the barrier and tell how many bytes are expected to come in (the transaction count)
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_data));
  } else {
    // 3b. Rest of threads just arrive
    token = bar.arrive();
  }

  // 3c. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_shared_to_global(data + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cde::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cde::cp_async_bulk_wait_group_read<0>();
  }
}