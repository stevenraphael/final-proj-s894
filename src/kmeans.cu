#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>



const int warp_size = 32;
const int block_size = 4;

__device__ void compute_clusters(
    int n,
    int k,
    int d,
    float *points,
    float *centroids,
    uint32_t *centroid_map
){
    int best_centroid = 0;
    float curr_dist = 0;

    

    int point_idx = threadIdx.x+threadIdx.y*warp_size+threadIdx.z*block_size+blockIdx.x*warp_size*block_size;
    for(int idx=0;idx<d;idx++){
        float point_coord = points[point_idx*d+idx];
        float centroid_coord = centroids[idx];
        curr_dist += (point_coord-centroid_coord)*(point_coord-centroid_coord);
    }
    for(int i=1;i<k;i++){    
        for(int idx=0;idx<d;idx++){
            float next_dist = 0;
            float point_coord = points[point_idx*d+idx];
            float centroid_coord = centroids[i*k+idx];
            next_dist += (point_coord-centroid_coord)*(point_coord-centroid_coord);
        }
        if(next_dist<curr_dist){
            curr_dist = next_dist;
            best_centroid = i;
        }
    }

    centroid_map[point_idx] = best_centroid;
}


const int points_per_thread = 1;

__device__ void compute_centroids(
    int n,
    int k,
    int d,
    float *points,
    float *centroids,
    uint32_t *centroid_map,
    float *local_dist_sums,
    float *local_point_counts,
    float *global_dist_sums,
    float *global_point_counts
){
}



__global__ void kmeans(
    int n,
    
    int k,
    int d,
    float *points,
    float *centroids,
    uint32_t *centroid_map,
){
}

struct Scene {
    int32_t dims;
    int32_t n_points;
    int32_t n_centroids;
    std::vector<float> features;
};


Scene gen_random(Rng &rng, int32_t dims, int32_t n_points, int32_t n_centroids){
    auto unif_100 = std::uniform_real_distribution<float>(-100.0f, 100.0f);
    auto true_centroids = std::vector<float>();

    const float stddev = 10.0;

    for (int32_t i = 0; i < n_centroids*dims; i++) {
        float z;
        z = unif_0_1(rng);

    
        // float z = std::max(unif_0_1(rng), unif_0_1(rng));
        true_centroids.push_back(z);
    }

    auto normal = std::normal_distribution<double>(0.0, stddev);

    auto features = std::vector<float>();

    for (int32_t cent = 0; cent < n_centroids; cent++) {
        for(int point=0;point<1+n_points/n_centroids;point++){
            if((1+n_points/n_centroids)*cent+point>=n_points){
                break;
            }
            for(int dim=0;dim<dims;dim++){
                float feature = normal(rng)+true_centroids[cent*dims+dim];
                features.push_back(feature);
            }
        }
    }

    auto scene = Scene{n_points, n_centroids, dims, features};

}