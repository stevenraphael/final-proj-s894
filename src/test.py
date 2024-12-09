import numpy
from libKMCUDA import kmeans_cuda
import time

print("____________________________________________________________________________________________")
numpy.random.seed(0)
arr = numpy.empty((10000, 2), dtype=numpy.float32)
arr[:2500] = numpy.random.rand(2500, 2) + [0, 2]
arr[2500:5000] = numpy.random.rand(2500, 2) - [0, 2]
arr[5000:7500] = numpy.random.rand(2500, 2) + [2, 0]
arr[7500:] = numpy.random.rand(2500, 2) - [2, 0]
start_time = time.time()
centroids, assignments = kmeans_cuda(arr, 4, verbosity=1, seed=3)
elapsed_time = time.time() - start_time
print(centroids)
print(f"Execution time: {elapsed_time:.2f} seconds")
