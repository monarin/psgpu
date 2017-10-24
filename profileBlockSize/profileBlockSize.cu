#include <stdio.h>
#include <cuda_profiler_api.h>

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

__global__ void kernel(short *a, int offset, short *dark, int offsetDark)
{
  int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
  int iDark = offsetDark + threadIdx.x + blockIdx.x*blockDim.x;
  a[i] -= dark[iDark];
}

/* ---------------------- host code -----------------------------*/
void fill( short *p, int n, int val ) {
  for(int i = 0; i < n; i++){
    p[i] = val;
  }
}

float maxError(short *a, int n)
{
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > maxE) maxE = error;
  }
  return maxE;
}

int main(int argc, char **argv)
{
  const int nPixels = 2400000;
  const int nEvents = atoi(argv[1]);
  const int n = nPixels * nEvents;
  
  const int nStreams = atoi(argv[2]);
  const int streamSize = n / nStreams;

  const int streamBytes = streamSize * sizeof(short);
  const int bytes = n * sizeof(short);
  const int darkBytes = nPixels * sizeof(short);

  // max block size is 1024
  const int blockSize = atoi(argv[3]);
  printf("Running with nStreams: %d streamSize: %d\n", nStreams, streamSize);
  int gridSize = streamSize / blockSize;
  printf("blockSize: %d gridSize: %d\n", blockSize, gridSize);

  int devId = 0;
  if (argc > 4) devId = atoi(argv[4]);
  
  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );

  // allocate pinned host memory and device memory
  short *a, *d_a; // data
  checkCuda( cudaMallocHost((void**)&a, bytes) ); // host pinned
  checkCuda( cudaMalloc((void**)&d_a, bytes) ); // device  
  short *dark, *d_dark; // dark
  checkCuda( cudaMallocHost((void**)&dark, darkBytes) ); 
  checkCuda( cudaMalloc((void**)&d_dark, darkBytes) ); 
  
  // prepare raw and dark data
  fill(a, n, 1);
  fill(dark, nPixels, 0);
  printf("Input values (Data): %d %d %d...%d %d %d\n", a[0], a[1], a[2], a[n-3], a[n-2], a[n-1]);
  printf("Input values (Dark): %d %d %d...%d %d %d\n", dark[0], dark[1], dark[2], dark[nPixels-3], dark[nPixels-2], dark[nPixels-1]);

  // serial copy for one dark 
  checkCuda( cudaMemcpy(d_dark, dark, darkBytes, cudaMemcpyHostToDevice) );

  float ms; // elapsed time in milliseconds

  // create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  checkCuda( cudaEventCreate(&dummyEvent) );
  for (int i = 0; i < nStreams; ++i)
    checkCuda( cudaStreamCreate(&stream[i]) );

  // asynchronous version 1: loop over {copy, kernel, copy}
  checkCuda( cudaEventRecord(startEvent, 0) );
  cudaProfilerStart();
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    int offsetDark = offset % nPixels;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset],
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );
    kernel<<<gridSize, blockSize, 0, stream[i]>>>(d_a, offset, d_dark, offsetDark);
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset],
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  cudaProfilerStop(); 
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n)); 
  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaEventDestroy(dummyEvent) );
  for (int i = 0; i < nStreams; ++i)
    checkCuda( cudaStreamDestroy(stream[i]) );
  cudaFree(d_a);
  cudaFreeHost(a);
  cudaFree(d_dark);
  cudaFreeHost(dark);

  return 0;
}
