#!/usr/bin/env python

from numba import jit
from numpy import arange
from numba.cuda import profile_start, profile_stop
import time

@jit
def sum2d(arr):
  M, N = arr.shape
  result = 0.0
  for i in range(M):
    for j in range(N):
      result += arr[i,j]

  return result

@jit
def sort_inplace(arr):
  arr.sort()

#a = arange(10000).reshape(100,100)
a = arange(10000)

start = time.time()
profile_start()
#res= sum2d(a)
sort_inplace(a)
profile_stop()
end = time.time()

print end-start
