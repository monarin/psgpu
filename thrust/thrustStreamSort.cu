#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

using namespace std;

#define NUM_THREADS 32
#define NUM_BLOCKS 16
#define NUM_STREAMS 32

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

struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const {
            return y - a * x;
        }
};

struct is_less
{
    __host__ __device__
    bool operator()(const float x)
    {
        return x < 100.0;
    }
};


int main()
{
    const int n = 71780 * NUM_STREAMS;
    const int bytes = n * sizeof(float);          // total size (bytes)
    const int darkBytes = n * sizeof(float);    // dark size (bytes)
    
    // allocate pinned host memory and device memory
    // RAW * nEVents
    float *a, *d_a, *a_out;                       // data address
    checkCuda( cudaMallocHost((void**)&a, bytes) );       // host pinned
    checkCuda( cudaMallocHost((void**)&a_out, bytes) );       // host pinned
    checkCuda( cudaMalloc((void**)&d_a, bytes) );         // device
    // SINGLE RAW
    float *raw;                                               // data address
    checkCuda( cudaMallocHost((void**)&raw, darkBytes) );     // host pinned
    // RAW-PEDESTAL
    float *pedCorrected, *d_pedCorrected;             // data address
    checkCuda( cudaMallocHost((void**)&pedCorrected, darkBytes) ); // host pinned
    checkCuda( cudaMalloc((void**)&d_pedCorrected, darkBytes) );  // device  
    // PEDESTAL
    float *dark, *d_dark;                     // dark address
    checkCuda( cudaMallocHost((void**)&dark, darkBytes) );    // host pinned
    checkCuda( cudaMalloc((void**)&d_dark, darkBytes) );      // device
    // PER-PIXEL GAIN
    float *gain, *d_gain;                     // dark address
    checkCuda( cudaMallocHost((void**)&gain, darkBytes) );    // host pinned
    checkCuda( cudaMalloc((void**)&d_gain, darkBytes) );      // device
    // RAW-PEDESTAL
    float *calib, *d_calib;                       // dark address
    checkCuda( cudaMallocHost((void**)&calib, darkBytes) );   // host pinned
    checkCuda( cudaMalloc((void**)&d_calib, darkBytes) );     // device  
    
    //load the text file and put it into a single string:
    ifstream inR("/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_raw_95.txt");
    ifstream inPC("/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_pedCorrected_95.txt");
    ifstream inP("/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_pedestal_95.txt");
    ifstream inG("/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_gain_95.txt");
    ifstream inC("/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_calib_95.txt");
    
    // Fill arrays from text files
    string line;
    for (unsigned int i = 0; i < n; i++){
      getline(inR, line);
      a[i] = atof(line.c_str());
      //populate all events with the same set of test data
      /*for (int j=0; j<nEvents; j++) {
        int offset = j * nPixels;
        a[offset + i] = raw[i];
      }*/
      getline(inPC, line);
      pedCorrected[i] = atof(line.c_str());
      getline(inP, line);
      dark[i] = atof(line.c_str());
      getline(inG, line);
      gain[i] = atof(line.c_str());
      getline(inC, line);
      calib[i] = atof(line.c_str());
    }

    printf("Input values (Data): %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", a[0], a[1], a[2], a[n-3], a[n-2], a[n-1]);
    printf("Input values (Dark): %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", dark[0], dark[1], dark[2], dark[n-3], dark[n-2], dark[n-1]);

    float ms; // elapsed time in milliseconds

    // create events and streams
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    cudaStream_t stream[NUM_STREAMS];
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    checkCuda( cudaEventCreate(&dummyEvent) );
    for (int i = 0; i < NUM_STREAMS; ++i)
      checkCuda( cudaStreamCreate(&stream[i]) );
    
    int streamSize = n / NUM_STREAMS;
    size_t streamMemSize = n * sizeof(float) / NUM_STREAMS;

    // --- Set kernel launch configuration
    dim3 nThreads = dim3(NUM_THREADS, 1, 1);
    dim3 nBlocks = dim3(NUM_BLOCKS, 1, 1);
    dim3 subKernelBlock = dim3((int) ceil( (float) nBlocks.x / 2));

    // serial copy for one dark to device 
    checkCuda( cudaMemcpy(d_dark, dark, darkBytes, cudaMemcpyHostToDevice) );

    // asynchronous version 1: loop over {copy, kernel, copy}
    checkCuda( cudaEventRecord(startEvent, 0) );
    cudaProfilerStart();
    for (int i = 0; i < NUM_STREAMS; ++i) {
      int offset = i * streamSize;

      checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset],
                                 streamMemSize, cudaMemcpyHostToDevice,
                                 stream[i]) );

    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * streamSize;
        
        // subtract pedestal and multiply gain
        float A = 1.0;
        thrust::transform(thrust::cuda::par.on(stream[i]),
                thrust::device_pointer_cast(&d_dark[offset]),
                thrust::device_pointer_cast(&d_dark[offset]) + streamSize,
                thrust::device_pointer_cast(&d_a[offset]),
                thrust::device_pointer_cast(&d_a[offset]),
                saxpy_functor(A));
        
        // filter only pixels with value below a threshold
        thrust::device_vector<float> result(streamSize);
        thrust::device_vector<float>::iterator end = thrust::copy_if(
                thrust::cuda::par.on(stream[i]),
                thrust::device_pointer_cast(&d_a[offset]),
                thrust::device_pointer_cast(&d_a[offset]) + streamSize,
                result.begin(), is_less());
        int len = end - result.begin();

        // sort the filtered pixels
        thrust::sort(thrust::cuda::par.on(stream[i]),
                result.begin(), result.end());

        // apply common mode
        thrust::device_vector<float> multor(streamSize);
        thrust::fill(multor.begin(), multor.end(), 1);
        thrust::transform(thrust::cuda::par.on(stream[i]),
                multor.begin(), multor.end(),
                thrust::device_pointer_cast(&d_a[offset]),
                thrust::device_pointer_cast(&d_a[offset]),
                saxpy_functor(result[len / 2]));

        /*thrust::sort(thrust::cuda::par.on(stream[i]),
                thrust::device_pointer_cast(&d_a[offset]),
                thrust::device_pointer_cast(&d_a[offset + streamSize])); */
        
    }
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        
        int offset = i * streamSize;
        checkCuda( cudaMemcpyAsync(&a_out[offset], &d_a[offset],
                                 streamMemSize, cudaMemcpyDeviceToHost,
                                 stream[i]) );

    
    }

    cudaProfilerStop(); 
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("GPU Calculation\n");
    printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * streamSize;
        printf("Sector %d\n", i);
        printf("Raw: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", a[offset+0], a[offset+1], a[offset+2], a[offset+streamSize-3], a[offset+streamSize-2], a[offset+streamSize-1]);
        printf("Dark: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", dark[offset+0], dark[offset+1], dark[offset+2], dark[offset+streamSize-3], dark[offset+streamSize-2], dark[offset+streamSize-1]);
        printf("Pedestal Corrected: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", pedCorrected[offset+0], pedCorrected[offset+1], pedCorrected[offset+2], pedCorrected[offset+streamSize-3], pedCorrected[offset+streamSize-2], pedCorrected[offset+streamSize-1]);
        printf("Calibrated: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", calib[offset+0], calib[offset+1], calib[offset+2], calib[offset+streamSize-3], calib[offset+streamSize-2], calib[offset+streamSize-1]);
        printf("Output: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", a_out[offset+0], a_out[offset+1], a_out[offset+2], a_out[offset+streamSize-3], a_out[offset+streamSize-2], a_out[offset+streamSize-1]);
        printf("Diff: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", calib[offset+0]-a_out[offset+0], calib[offset+1]-a_out[offset+1], calib[offset+2]-a_out[offset+2], calib[offset+streamSize-3]-a_out[offset+streamSize-3], calib[offset+streamSize-2]-a_out[offset+streamSize-2], calib[offset+streamSize-1]-a_out[offset+streamSize-1]);
    }

    // cleanup
    checkCuda( cudaEventDestroy(startEvent) );
    checkCuda( cudaEventDestroy(stopEvent) );
    checkCuda( cudaEventDestroy(dummyEvent) );
    for (int i = 0; i < NUM_STREAMS; ++i)
      checkCuda( cudaStreamDestroy(stream[i]) );
    cudaFree(d_a);
    cudaFreeHost(a);
    cudaFree(d_dark);
    cudaFreeHost(dark);
    return 0;
}
