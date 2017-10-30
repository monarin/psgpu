#include <stdio.h>
#include <cuda_profiler_api.h>
#include <unistd.h>

#include <sys/time.h>
#include <iostream>
#include <iomanip>
using namespace std;

#include <string>
#include <sstream>
#include <fstream>
#define N_PIXELS 2296960

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

__global__ void kernel(float *a, int offset, float *dark, int streamSize)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x; // thread id
  
  // check if idx < streamSize
  if (idx < streamSize) {
    int iData = offset + idx;
    int iDark = iData % N_PIXELS;
    a[iData] -= dark[iDark];
  }
}

/* ---------------------- host code -----------------------------*/
void fill( float *p, int n, float val ) {
  for(int i = 0; i < n; i++){
    p[i] = val;
  }
}

float maxError(float *aCalc, float *aKnown, int nEvents, int nPixels)
{
  float maxE = 0;
  for (int i = 0; i < nEvents; i++) {
    int offset = i * nPixels;
    for (int j = 0; j < nPixels; j++) {
      int idx = offset + j;
      float error = fabs(aCalc[idx]-aKnown[j]);
      if (error > 1.0)
      printf("offset: %d j: %d idx: %d error %e aCalc[idx]: %8.2f aKnown[j]: %8.2f\n", offset, j, idx, error, aCalc[idx], aKnown[j]);
      if (error > maxE) maxE = error;
    }
  }
  return maxE;
}

void host_calc(float *a, float *dark, int nPixels) {
  // host calculation
  struct timeval start, end;

  long seconds, useconds;
  double mtime;

  gettimeofday(&start, NULL);

  for(int i=0; i < nPixels; i++)
    a[i] -= dark[i];

  gettimeofday(&end, NULL);

  seconds  = end.tv_sec  - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;
  mtime = ((seconds) * 1000000 + useconds)/1000.0;// + 0.5;

  cout << "Host dark-subtraction took "<< mtime <<" ms for 1 event."<< endl;
}

int main(int argc, char **argv)
{
  const int nPixels = N_PIXELS;			// no. of pixels per image
  const int nEvents = atoi(argv[1]);			// no. of events
  const int n = nPixels * nEvents;			// total number of pixels
  
  const int nStreams = atoi(argv[2]);			// no. of stream
  const int blockSize = atoi(argv[3]);                  // block size (max is 1024)

  const int bytes = n * sizeof(float);			// total size (bytes)
  const int darkBytes = nPixels * sizeof(float);	// dark size (bytes)
  
  int devId = 0;
  if (argc > 4) devId = atoi(argv[4]);			// device ID (optional)
  
  // print device name
  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );

  // allocate pinned host memory and device memory
  // RAW * nEVents
  float *a, *d_a; 						// data address
  checkCuda( cudaMallocHost((void**)&a, bytes) ); 		// host pinned
  checkCuda( cudaMalloc((void**)&d_a, bytes) ); 		// device
  // SINGLE RAW
  float *raw;                                               // data address
  checkCuda( cudaMallocHost((void**)&raw, darkBytes) );     // host pinned
  // RAW-PEDESTAL
  float *pedCorrected, *d_pedCorrected; 			// data address
  checkCuda( cudaMallocHost((void**)&pedCorrected, darkBytes) ); // host pinned
  checkCuda( cudaMalloc((void**)&d_pedCorrected, darkBytes) ); 	// device  
  // PEDESTAL
  float *dark, *d_dark;					 	// dark address
  checkCuda( cudaMallocHost((void**)&dark, darkBytes) ); 	// host pinned
  checkCuda( cudaMalloc((void**)&d_dark, darkBytes) );		// device
  // PER-PIXEL GAIN
  float *gain, *d_gain;					 	// dark address
  checkCuda( cudaMallocHost((void**)&gain, darkBytes) ); 	// host pinned
  checkCuda( cudaMalloc((void**)&d_gain, darkBytes) );		// device
  // RAW-PEDESTAL
  float *calib, *d_calib;					 	// dark address
  checkCuda( cudaMallocHost((void**)&calib, darkBytes) ); 	// host pinned
  checkCuda( cudaMalloc((void**)&d_calib, darkBytes) );		// device  
  
  //load the text file and put it into a single string:
  ifstream inR("/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_raw_95.txt");
  ifstream inPC("/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_pedCorrected_95.txt");
  ifstream inP("/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_pedestal_95.txt");
  ifstream inG("/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_gain_95.txt");
  ifstream inC("/reg/data/ana14/cxi/cxitut13/res/yoon82/psgpu/profileBlockSize/cxid9114_calib_95.txt");
  // Fill arrays from text files
  string line;
  for (unsigned int i=0; i<nPixels; i++){
    getline(inR, line);
    raw[i] = atof(line.c_str());
    //populate all events with the same set of test data
    for (int j=0; j<nEvents; j++) {
      int offset = j * nPixels;
      a[offset + i] = raw[i];
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

  printf("Input values (Data): %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", a[0], a[1], a[2], a[n-3], a[n-2], a[n-1]);
  printf("Input values (Dark): %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", dark[0], dark[1], dark[2], dark[nPixels-3], dark[nPixels-2], dark[nPixels-1]);

  // host calculation 
  host_calc(raw, dark, nPixels);
  printf("Host Calculation\n");
  printf("Input values (Data calc.): %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", raw[0], raw[1], raw[2], raw[nPixels-3], raw[nPixels-2], raw[nPixels-1]);
  printf("Input values (Data known): %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", pedCorrected[0], pedCorrected[1], pedCorrected[2], pedCorrected[nPixels-3], pedCorrected[nPixels-2], pedCorrected[nPixels-1]);
  printf("  max error: %e\n", maxError(raw, pedCorrected, 1, nPixels));

  // serial copy for one dark to device 
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
    int streamSize = ceil( (double) n / nStreams );  // stream size (pixels)
    int offset = i * streamSize;

    // check if last stream has full length
    if ( (i + 1) * streamSize > n ) streamSize = n - (i * streamSize);

    int streamBytes = streamSize * sizeof(float);   // stream size (bytes)
    //printf("Stream#: %d streamSize: %d offset=%d\n", i, streamSize, offset);
    int gridSize = ceil(  (double) streamSize / blockSize );               // grid size
    //printf("blockSize: %d gridSize: %d\n", blockSize, gridSize);

    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset],
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );
    kernel<<<gridSize, blockSize, 0, stream[i]>>>(d_a, offset, d_dark, streamSize);
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset],
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  cudaProfilerStop(); 
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("GPU Calculation\n");
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("Input values (Data calc.): %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", a[0], a[1], a[2], a[n-3], a[n-2], a[n-1]);
  printf("Input values (Data known): %8.2f %8.2f %8.2f...%8.2f %8.2f %8.2f\n", pedCorrected[0], pedCorrected[1], pedCorrected[2], pedCorrected[nPixels-3], pedCorrected[nPixels-2], pedCorrected[nPixels-1]);
  printf("  max error: %e\n", maxError(a, pedCorrected, nEvents, nPixels));

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
