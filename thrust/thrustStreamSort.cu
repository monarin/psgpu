#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
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


int main(int argc, char **argv)
{
    const int nPixels = 2296960;
    int nEvents = 1;
    if (argc > 1) nEvents = atoi(argv[1]);
    int nAlg = 1; // algorithm number
    if (argc > 2) nAlg = atoi(argv[2]);

    const int n = nPixels * nEvents;
    const int bytes = n * sizeof(float);          // total size (bytes)
    const int darkBytes = nPixels * sizeof(float);    // dark size (bytes)
    
    // allocate pinned host memory and device memory
    float *h_in, *h_out, *d_in;
    checkCuda( cudaMallocHost((void**)&h_in, bytes) );
    checkCuda( cudaMallocHost((void**)&h_out, bytes) );
    checkCuda( cudaMalloc((void**)&d_in, bytes) );

    // SINGLE RAW
    float *raw;                                               // data address
    checkCuda( cudaMallocHost((void**)&raw, darkBytes) );     // host pinned
    
    // RAW-PEDESTAL-CORRECTED
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
    
    // RAW-CALIBRATED
    float *calib, *d_calib;                       // dark address
    checkCuda( cudaMallocHost((void**)&calib, darkBytes) );   // host pinned
    checkCuda( cudaMalloc((void**)&d_calib, darkBytes) );     // device  
    
    //load the text file and put it into a single string:
    ifstream inR("/reg/neh/home/monarin/psgpu/data/cxid9114_raw_95.txt");
    ifstream inPC("/reg/neh/home/monarin/psgpu/data/cxid9114_pedCorrected_95.txt");
    ifstream inP("/reg/neh/home/monarin/psgpu/data/cxid9114_pedestal_95.txt");
    ifstream inG("/reg/neh/home/monarin/psgpu/data/cxid9114_gain_95.txt");
    ifstream inC("/reg/neh/home/monarin/psgpu/data/cxid9114_calib_95.txt");
    
    // Fill arrays from text files
    string line;
    for (unsigned int i = 0; i < nPixels; i++){
      getline(inR, line);
      raw[i] = atof(line.c_str());
      //populate all events with the same set of test data
      for (int j=0; j<nEvents; j++) {
        int offset = j * nPixels;
        h_in[offset + i] = raw[i];
      }
      getline(inPC, line);
      pedCorrected[i] = atof(line.c_str());
      getline(inP, line);
      dark[i] = atof(line.c_str());
      getline(inG, line);
      gain[i] = atof(line.c_str());
      getline(inC, line);
      calib[i] = atof(line.c_str());
    }
    

    printf("Data (single): %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", h_in[0], h_in[1], h_in[2], h_in[nPixels-3], h_in[nPixels-2], h_in[nPixels-1]);
    printf("Dark         : %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", dark[0], dark[1], dark[2], dark[nPixels-3], dark[nPixels-2], dark[nPixels-1]);

    float ms; // elapsed time in milliseconds

    // create events and streams
    cudaEvent_t startEvent, startStreamEvent, copyEvent, pedestalEvent, filterEvent, sortEvent, commonEvent, endStreamEvent, stopEvent;
    cudaStream_t stream[NUM_STREAMS];
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&startStreamEvent) );
    checkCuda( cudaEventCreate(&copyEvent) );
    checkCuda( cudaEventCreate(&pedestalEvent) );
    checkCuda( cudaEventCreate(&filterEvent) );
    checkCuda( cudaEventCreate(&sortEvent) );
    checkCuda( cudaEventCreate(&commonEvent) );
    checkCuda( cudaEventCreate(&endStreamEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    for (int i = 0; i < NUM_STREAMS; ++i)
      checkCuda( cudaStreamCreate(&stream[i]) );
    
    const int streamSize = 185 * 388; // streamSize is always a sector
    const size_t streamMemSize = streamSize * sizeof(float);

    // --- Set kernel launch configuration
    dim3 nThreads = dim3(NUM_THREADS, 1, 1);
    dim3 nBlocks = dim3(NUM_BLOCKS, 1, 1);
    dim3 subKernelBlock = dim3((int) ceil( (float) nBlocks.x / 2));

    // serial copy for one dark to device 
    checkCuda( cudaMemcpy(d_dark, dark, darkBytes, cudaMemcpyHostToDevice) );

    checkCuda( cudaEventRecord(startEvent, 0) );
    cudaProfilerStart();

    // setup thrust device pointer
    thrust::device_ptr<float> d_in_ptr = thrust::device_pointer_cast(d_in);
    thrust::device_ptr<float> d_dark_ptr = thrust::device_pointer_cast(d_dark);
    
    if (nAlg == 1) {
      
      // asynchronous version 1: loop over {copy, kernel, copy}
      for (int i = 0; i < NUM_STREAMS; i++) {

          // each stream is working on a sector for all events
          for (int j = 0; j < nEvents; j++) {

            checkCuda( cudaEventRecord(startStreamEvent, 0) );
            
            int offset = (j * nPixels) + (i * streamSize);
            int offsetDark = i * streamSize;

            checkCuda( cudaMemcpyAsync(&d_in[offset], &h_in[offset],
                                   streamMemSize, cudaMemcpyHostToDevice,
                                   stream[i]) );
            checkCuda( cudaEventRecord(copyEvent, 0) );
                                   
            // subtract pedestal and multiply gain
            thrust::transform(thrust::cuda::par.on(stream[i]),
                  d_dark_ptr + offsetDark, d_dark_ptr + offsetDark + streamSize,
                  d_in_ptr + offset, d_in_ptr + offset,
                  saxpy_functor(1.0));
            checkCuda( cudaEventRecord(pedestalEvent, 0) );
          
            // filter only pixels with value below a threshold
            thrust::device_vector<float> result(streamSize);
            
            thrust::device_vector<float>::iterator end = thrust::copy_if(
                  thrust::cuda::par.on(stream[i]),
                  d_in_ptr + offset,
                  d_in_ptr + offset + streamSize,
                  result.begin(), is_less());
            int len = end - result.begin();
            checkCuda( cudaEventRecord(filterEvent, 0) );

            // sort the filtered pixels
            thrust::sort(thrust::cuda::par.on(stream[i]), result.begin(), result.end());
            checkCuda( cudaEventRecord(sortEvent, 0) );

            // apply common mode
            thrust::device_vector<float> multor(streamSize);
            thrust::fill(multor.begin(), multor.end(), 1);
            thrust::transform(thrust::cuda::par.on(stream[i]),
                  multor.begin(), multor.end(),
                  d_in_ptr + offset,
                  d_in_ptr + offset,
                  saxpy_functor(result[len / 2]));
            checkCuda( cudaEventRecord(commonEvent, 0) );
            
            // copy data out
            checkCuda( cudaMemcpyAsync(&h_out[offset], &d_in[offset],
                                   streamMemSize, cudaMemcpyDeviceToHost,
                                   stream[i]) );
       
            checkCuda( cudaEventRecord(endStreamEvent, 0) );
            
          }
      }  
    } else {
      
      // asynchronous version 2:  copy {loop over}, kernel {loop over}, copy {loo over}
      for (int i = 0; i < NUM_STREAMS; i++) {

          // each stream is working on a sector for all events
          for (int j = 0; j < nEvents; j++) {

            int offset = (j * nPixels) + (i * streamSize);
            
            checkCuda( cudaEventRecord(startStreamEvent, 0) );

            checkCuda( cudaMemcpyAsync(&d_in[offset], &h_in[offset],
                                   streamMemSize, cudaMemcpyHostToDevice,
                                   stream[i]) );
            
            checkCuda( cudaEventRecord(copyEvent, 0) );
          }
      }
      
      for (int i = 0; i < NUM_STREAMS; i++) {

          // each stream is working on a sector for all events
          for (int j = 0; j < nEvents; j++) {

            int offset = (j * nPixels) + (i * streamSize);
            int offsetDark = i * streamSize;

            // subtract pedestal and multiply gain
            thrust::transform(thrust::cuda::par.on(stream[i]),
                  thrust::device_pointer_cast(&d_dark[offsetDark]),
                  thrust::device_pointer_cast(&d_dark[offsetDark]) + streamSize,
                  thrust::device_pointer_cast(&d_in[offset]),
                  thrust::device_pointer_cast(&d_in[offset]),
                  saxpy_functor(1.0));
            checkCuda( cudaEventRecord(pedestalEvent, 0) );
          
            // filter only pixels with value below a threshold
            thrust::device_vector<float> result(streamSize);
            thrust::device_vector<float>::iterator end = thrust::copy_if(
                  thrust::cuda::par.on(stream[i]),
                  thrust::device_pointer_cast(&d_in[offset]),
                  thrust::device_pointer_cast(&d_in[offset]) + streamSize,
                  result.begin(), is_less());
            int len = end - result.begin();
            checkCuda( cudaEventRecord(filterEvent, 0) );

            // sort the filtered pixels
            thrust::sort(thrust::cuda::par.on(stream[i]),
                  result.begin(), result.end());
            checkCuda( cudaEventRecord(sortEvent, 0) );

            // apply common mode
            thrust::device_vector<float> multor(streamSize);
            thrust::fill(multor.begin(), multor.end(), 1);
            thrust::transform(thrust::cuda::par.on(stream[i]),
                  multor.begin(), multor.end(),
                  thrust::device_pointer_cast(&d_in[offset]),
                  thrust::device_pointer_cast(&d_in[offset]),
                  saxpy_functor(result[len / 2]));
            checkCuda( cudaEventRecord(commonEvent, 0) );
            
            

          }
      }
      
      for (int i = 0; i < NUM_STREAMS; i++) {

          // each stream is working on a sector for all events
          for (int j = 0; j < nEvents; j++) {

            int offset = (j * nPixels) + (i * streamSize);
            
            checkCuda( cudaMemcpyAsync(&h_out[offset], &d_in[offset],
                                   streamMemSize, cudaMemcpyDeviceToHost,
                                   stream[i]) );
          }
      }
    
    }
    

    cudaProfilerStop(); 
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    float copyMs, pedestalMs, filterMs, sortMs, commonMs; 
    checkCuda( cudaEventElapsedTime(&copyMs, startStreamEvent, copyEvent) );
    checkCuda( cudaEventElapsedTime(&pedestalMs, copyEvent, pedestalEvent) );
    checkCuda( cudaEventElapsedTime(&filterMs, pedestalEvent, filterEvent) );
    checkCuda( cudaEventElapsedTime(&sortMs, filterEvent, sortEvent) );
    checkCuda( cudaEventElapsedTime(&commonMs, sortEvent, commonEvent) );
    printf("GPU Calculation\n");
    printf("Total Time (ms): %f\n", ms);
    printf("  Copy Time (ms): %f\n", copyMs);
    printf("  Pedestal Time (ms): %f\n", pedestalMs);
    printf("  Filter Time (ms): %f\n", filterMs);
    printf("  Sort Time (ms): %f\n", sortMs);
    printf("  Common Mode Time (ms): %f\n", commonMs);
   
    /* 
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * streamSize;
        printf("Sector %d\n", i);
        printf("Raw: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", h_in[offset+0], h_in[offset+1], h_in[offset+2], h_in[offset+streamSize-3], h_in[offset+streamSize-2], h_in[offset+streamSize-1]);
        printf("Dark: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", dark[offset+0], dark[offset+1], dark[offset+2], dark[offset+streamSize-3], dark[offset+streamSize-2], dark[offset+streamSize-1]);
        printf("Pedestal Corrected: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", pedCorrected[offset+0], pedCorrected[offset+1], pedCorrected[offset+2], pedCorrected[offset+streamSize-3], pedCorrected[offset+streamSize-2], pedCorrected[offset+streamSize-1]);
        printf("Calibrated: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", calib[offset+0], calib[offset+1], calib[offset+2], calib[offset+streamSize-3], calib[offset+streamSize-2], calib[offset+streamSize-1]);
        printf("Output: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", h_out[offset+0], h_out[offset+1], h_out[offset+2], h_out[offset+streamSize-3], h_out[offset+streamSize-2], h_out[offset+streamSize-1]);
        printf("Diff: %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", calib[offset+0]-h_out[offset+0], calib[offset+1]-h_out[offset+1], calib[offset+2]-h_out[offset+2], calib[offset+streamSize-3]-h_out[offset+streamSize-3], calib[offset+streamSize-2]-h_out[offset+streamSize-2], calib[offset+streamSize-1]-h_out[offset+streamSize-1]);
    }
    */
    // cleanup
    checkCuda( cudaEventDestroy(startEvent) );
    checkCuda( cudaEventDestroy(startStreamEvent) );
    checkCuda( cudaEventDestroy(copyEvent) );
    checkCuda( cudaEventDestroy(pedestalEvent) );
    checkCuda( cudaEventDestroy(filterEvent) );
    checkCuda( cudaEventDestroy(sortEvent) );
    checkCuda( cudaEventDestroy(commonEvent) );
    checkCuda( cudaEventDestroy(endStreamEvent) );
    checkCuda( cudaEventDestroy(stopEvent) );

    for (int i = 0; i < NUM_STREAMS; ++i)
      checkCuda( cudaStreamDestroy(stream[i]) );
    
    cudaFree(d_in);
    cudaFreeHost(h_in);
    cudaFree(d_dark);
    cudaFreeHost(dark);
    return 0;
}
