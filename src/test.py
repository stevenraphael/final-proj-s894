import numpy as np
from libKMCUDA import kmeans_cuda
import time

# Parameters
dims = 10  # Number of dimensions
n_points = 1000000  # Total number of points
n_centroids = 50  # Number of centroids
stddev = 10.0  # Standard deviation for point distribution around centroids

# Set seed for reproducibility
np.random.seed(0)

# Generate true centroids uniformly in the range [-100, 100]
true_centroids = np.random.uniform(-100.0, 100.0, size=(n_centroids, dims))

# Generate points around centroids
points = []
points_per_centroid = n_points // n_centroids
for centroid in true_centroids:
    cluster_points = np.random.normal(loc=centroid, scale=stddev, size=(points_per_centroid, dims))
    points.append(cluster_points)

# Handle any leftover points due to rounding
remaining_points = n_points - (points_per_centroid * n_centroids)
if remaining_points > 0:
    extra_points = np.random.normal(loc=true_centroids[0], scale=stddev, size=(remaining_points, dims))
    points.append(extra_points)

# Combine all points into a single array
arr = np.vstack(points).astype(np.float32)

# Run kmeans_cuda
print("____________________________________________________________________________________________")
min_elapsed = 1000000.0

for i in range(10):
    start_time = time.time()
    centroids, assignments = kmeans_cuda(arr, n_centroids, verbosity=1, seed=3)
    elapsed_time = (time.time() - start_time)*1000
    min_elapsed = min(min_elapsed, elapsed_time)

# Print the results
print("Generated Centroids (True Centroids):")
print(true_centroids)
print("\nKMeans CUDA Centroids:")
print(centroids)
print(f"\nExecution time: {min_elapsed:.2f} milliseconds")
