#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "Utilities.cuh"

using namespace std;

#define NUM_THREADS 32
#define NUM_BLOCKS 16
#define NUM_STREAMS 3

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
      if (result != cudaSuccess) {
              fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
                  assert(result == cudaSuccess);
                    }
#endif
        return result;
}

struct BinaryOp{ __host__ __device__ int operator()(const int& o1, const int& o2) { return o1 * o2; } };

void fill(int *a, int N, int val)
{
    for (int i = 0; i < N; i++) {
        a[i] = val;
    }
}

int main()
{
    const int N = 6000000;

    // --- Host side
    int *h_in = new int[N]; 
    fill(h_in, N, 5);
    checkCuda(cudaHostRegister(h_in, N * sizeof(int), cudaHostRegisterPortable));

    int *h_out = new int[N];
    fill(h_out, N, 0);
    checkCuda(cudaHostRegister(h_out, N * sizeof(int), cudaHostRegisterPortable));

    int *h_checkResults = new int[N];
    fill(h_checkResults, N, 25);

    // --- Device side
    int *d_in, *d_out;
    checkCuda(cudaMalloc((void **)&d_in, N * sizeof(int)));
    checkCuda(cudaMalloc((void **)&d_out, N * sizeof(int)));

    int streamSize = N / NUM_STREAMS;
    size_t streamMemSize = N * sizeof(int) / NUM_STREAMS;

    // --- Set kernel launch configuration
    dim3 nThreads = dim3(NUM_THREADS, 1, 1);
    dim3 nBlocks = dim3(NUM_BLOCKS, 1, 1);
    dim3 subKernelBlock = dim3((int) ceil( (float) nBlocks.x / 2));

    // --- Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++)
        checkCuda(cudaStreamCreate(&streams[i]));

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_in[offset], &h_in[offset], streamMemSize, cudaMemcpyHostToDevice, streams[i]);

        printf("Input: %d %d %d...%d %d %d\n", h_in[offset+0], h_in[offset+1], h_in[offset+2], h_in[offset+streamSize-3], h_in[offset+streamSize-2], h_in[offset+streamSize-1]);
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * streamSize;

        thrust::transform(thrust::cuda::par.on(streams[i]),
                    thrust::device_pointer_cast(&d_in[offset]), 
                    thrust::device_pointer_cast(&d_in[offset]) + streamSize / 2,
                    thrust::device_pointer_cast(&d_in[offset]),
                    thrust::device_pointer_cast(&d_out[offset]), 
                    BinaryOp());

        thrust::transform(thrust::cuda::par.on(streams[i]), 
                    thrust::device_pointer_cast(&d_in[offset + streamSize / 2]),
                    thrust::device_pointer_cast(&d_in[offset + streamSize / 2]) + streamSize / 2,
                    thrust::device_pointer_cast(&d_in[offset + streamSize / 2]),
                    thrust::device_pointer_cast(&d_out[offset + streamSize / 2]),
                    BinaryOp());
    }

    // copy data out
    for (int i = 0; i < NUM_STREAMS; i ++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&h_out[offset], &d_out[offset], streamMemSize, cudaMemcpyDeviceToHost, streams[i]);

        printf("Output: %d %d %d...%d %d %d\n", h_out[offset+0], h_out[offset+1], h_out[offset+2], h_out[offset+streamSize-3], h_out[offset+streamSize-2], h_out[offset+streamSize-1]);
    }
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCuda(cudaStreamSynchronize(streams[i]));
    }

    checkCuda(cudaDeviceSynchronize());
    
    // -- Release resources
    checkCuda(cudaHostUnregister(h_in));
    checkCuda(cudaHostUnregister(h_out));
    checkCuda(cudaFree(d_in));
    checkCuda(cudaFree(d_out));

    for (int i = 0; i < NUM_STREAMS; i++)
        checkCuda(cudaStreamDestroy(streams[i]));

    cudaDeviceReset();

    // -- GPU output check
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += h_checkResults[i] - h_out[i];
        if ( i < 10 ) printf("%d %d", h_checkResults[i], h_out[i]);
    }

    cout << "Error between CPU and GPU: " << sum << endl;

    delete[] h_in;
    delete[] h_out;
    delete[] h_checkResults;

    return 0;
}
