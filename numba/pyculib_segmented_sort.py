from __future__ import print_function
from timeit import default_timer as time
import pyculib
import numpy as np

keys = np.random.random((185*355,))
print(keys)
vals = np.arange(keys.size, dtype=np.int32)
segments = np.array([len(keys)], dtype=np.int32)
pyculib.numba.cuda.profile_start()
stream = pyculib.cuda.stream()
s = time()
pyculib.sorting.segmented_sort(keys, vals, segments, stream)
e = time()
pyculib.numba.cuda.profile_stop()
print("time: (ms)", (e - s)/1000)
print(keys)

