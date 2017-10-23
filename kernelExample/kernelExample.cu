#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

//#define N 1000000
#define SQRT_TWO_PI 2.506628274631000
#define BLOCK_D1 1024
#define BLOCK_D2 1
#define BLOCK_D3 1

// Note: Needs compute capability >= 2.0 for calculation with doubles, so compile with:
// nvcc kernelExample.cu -arch=compute_20 -code=sm_20,compute_20 -o kernelExample
// -use_fast_math doesn't seem to have any effect on speed

// CUDA kernel:
__global__ void calc_calib(short* raws, short* darks, int N) {
  // note that this assumes no third dimension to the grid
  // id of the block
  int myblock = blockIdx.x + blockIdx.y * gridDim.x;
  // size of each block (within grid of blocks)
  int blocksize = blockDim.x * blockDim.y * blockDim.z;
  // id of thread in a given block
  int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
  // assign overall id/index of the thread
  int idx = myblock * blocksize + subthread;

  if(idx < N) {
    raws[idx] -= darks[idx];
  }
}

// CPU analog for speed comparison
int calc_calib_cpu(short* raws, short* darks, int N) {
  for(int idx = 0; idx < N; idx++){
    raws[idx] -= darks[idx];
  }
  return 0;
}

/* ---------------------- host code -----------------------------*/
void fill( short *p, int n, int val ) {
  for(int i = 0; i < n; i++){
    p[i] = val;
  } 
}

double read_timer() {
  struct timeval end;
  gettimeofday( &end, NULL);
  return end.tv_sec+1.e-6*end.tv_usec;
}

int main (int argc, char *argv[]) {
  short* cpu_raws;
  short* gpu_raws;
  short* cpu_darks;
  short* gpu_darks;
  int N;
  cudaError_t cudaStat;

  printf("==========================================\n");
  for( N = 2296960; N <= 2296960; N+=2296960 ) {
    cpu_raws = (short*) malloc( sizeof(short)*N );
    cudaStat = cudaMalloc(&gpu_raws, sizeof(short)*N);
    if(cudaStat != cudaSuccess) {
      printf ("device memory allocation failed on gpu_raws");
      return EXIT_FAILURE;
    }
    cpu_darks = (short*) malloc( sizeof(short)*N );
    cudaStat = cudaMalloc(&gpu_darks, sizeof(short)*N);
    if(cudaStat != cudaSuccess) {
      printf ("device memory allocation failed on gpu_darks");
      return EXIT_FAILURE;
    }

    // fixed block dimensions (1024x1x1 threads)
    const dim3 blockSize(BLOCK_D1, BLOCK_D2, BLOCK_D3);

    // determine number of blocks we need for a given problem size
    int tmp = ceil(pow(N/(BLOCK_D1 * BLOCK_D2 * BLOCK_D3), 0.5));
    printf("Grid dimension is %i x %i\n", tmp, tmp);
    dim3 gridSize(tmp, tmp, 1);

    int nthreads = BLOCK_D1*BLOCK_D2*BLOCK_D3*tmp*tmp;
    if (nthreads < N){
      printf("\n================ NOT ENOUGH THREADS TO COVER N=%d =======================\n\n", N);
    } else {
      printf("Launching %d threads (N=%d)\n", nthreads, N);
    }

    // simulate 'data'
    fill(cpu_raws, N, 3);
    fill(cpu_darks, N, 1);
    printf("Input values (raw): %d %d %d...\n", cpu_raws[0], cpu_raws[1], cpu_raws[2]);
    printf("Input values (dark): %d %d %d...\n", cpu_darks[0], cpu_darks[1], cpu_darks[2]);
    cudaDeviceSynchronize();
    double tInit = read_timer();

    // copy input data to the GPU
    cudaStat = cudaMemcpy(gpu_raws, cpu_raws, N*sizeof(short), cudaMemcpyHostToDevice);
    printf("Memory Copy from Host to Device (raw)");
    if (cudaStat){
      printf("failed.\n");
    } else {
      printf("successful.\n");
    }
    cudaStat = cudaMemcpy(gpu_darks, cpu_darks, N*sizeof(short), cudaMemcpyHostToDevice);
    printf("Memory Copy from Host to Device (dark)");
    if (cudaStat){
      printf("failed.\n");
    } else {
      printf("successful.\n");
    }
    cudaDeviceSynchronize();
    double tTransferToGPU = read_timer();

    // do the calculation
    calc_calib<<<gridSize, blockSize>>>(gpu_raws, gpu_darks, N);

    cudaDeviceSynchronize();
    double tCalc = read_timer();

    cudaStat = cudaMemcpy(cpu_raws, gpu_raws, N*sizeof(short), cudaMemcpyDeviceToHost);
    printf("Memory Copy from Device to Host (raw) ");
    if (cudaStat){
      printf("failed.\n");
    } else {
      printf("successful.\n");
    }
    cudaDeviceSynchronize();
    double tTransferFromGPU = read_timer();

    printf("Output values: %d %d %d...%d %d %d\n", cpu_raws[0], cpu_raws[1], cpu_raws[2], cpu_raws[N-3], cpu_raws[N-2], cpu_raws[N-1]);

    // do calculation on CPU for comparison (unfair as this will only use one core)
    fill(cpu_raws, N, 3);
    fill(cpu_darks, N, 1);
    double tInit2 = read_timer();
    calc_calib_cpu(cpu_raws, cpu_darks, N);
    double tCalcCPU = read_timer();

    printf("Output values (CPU): %d %d %d...\n", cpu_raws[0], cpu_raws[1], cpu_raws[2]);

    printf("Timing results for n = %d\n", N);
    printf("Transfer to GPU time: %f\n", tTransferToGPU - tInit);
    printf("Calculation time (GPU): %f\n", tCalc - tTransferToGPU);
    printf("Calculation time (CPU): %f\n", tCalcCPU - tInit2);
    printf("Transfer from GPU time: %f\n", tTransferFromGPU - tCalc);

    printf("Freeing memory...\n");
    printf("==============================================\n");
    free(cpu_raws);
    free(cpu_darks);
    cudaFree(gpu_raws);
    cudaFree(gpu_darks);

  }
  printf("\n\nFinished.\n\n");
  return 0;

}





