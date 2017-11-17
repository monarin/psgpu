from numba import jit
import numpy as np
import time

@jit
def hello(data):
      data[cuda.blockIdx.x, cuda.threadIdx.x] = cuda.blockIdx.x

numBlocks = 5
threadsPerBlock = 10

data = np.ones((numBlocks, threadsPerBlock), dtype=np.uint8)

hello[numBlocks, threadsPerBlock](data)

print(data)

numba.cuda.profile_stop()
