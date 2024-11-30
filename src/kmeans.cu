#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <cuda/std/unordered_map>


enum class Mode {
    TEST,
    BENCHMARK,
};


struct Scene {
    int32_t dims;
    int32_t n_points;
    int32_t n_centroids;
    std::vector<float> true_centroids;
    std::vector<float> initial_centroids;
    std::vector<float> features;
};

namespace kmeans {


const int warp_size = 32;
const int block_size = 4;

size_t get_workspace_size(size_t n) {
    return n;
}

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

    

    int point_idx = threadIdx.x+threadIdx.y*warp_size+blockIdx.x*warp_size*block_size;
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


const int points_per_thread = 8;

const int warp_size_2 = 4;

const int block_size_2 = 32;


const int MAX_CENTROIDS = 100;

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
    int dim = threadIdx.x;
    int point_idx = threadIdx.y*points_per_thread+threadIdx.z*points_per_thread*warp_size_2
                    +blockIdx.x*points_per_thread*warp_size_2*block_size_2;

    cuda::std::unordered_map<int, float> sum_map;
    cuda::std::unordered_map<int, int> count_map;

    for(int p=point_idx;p<points_per_thread;p++){
        int label = centroid_map[p];
        if(count_map.contains(centroid_map[p])){
            count_map[p]++;
            centroid_map[p] += points[p*d+dim];
        }
        else{
            count_map[p]=1;
            centroid_map[p] = points[p*d+dim];
        }
    }
    
}



__global__ void kmeans(
    int n,
    int k,
    int d,
    float *points,
    float *initial_centroids,
    float *output_centroids,
    uint32_t *centroid_map,
){
}


void launch_kmeans(
    int n,
    int k,
    int d,
    float *points,
    float *initial_centroids,
    float *output_centroids,
    GpuMemoryPool &memory_pool
){

}


}


struct Results {
    float average_squared_dist;
    std::vector<float> centroids;
    double time_ms;
};


Results run_config(Mode mode, Scene const &scene) {
    auto points_gpu = GpuBuf<float>(scene.features);
    auto centroids_gpu = GpuBuf<float>(scene.features);
    auto memory_pool = GpuMemoryPool();


    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(points_gpu.data, 0, scene.features.size() * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(centroids_gpu.data, 0, scene.true_centroids.size() * sizeof(float)));

        memory_pool.reset();
    };

    auto f = [&]() {
        kmeans::launch_render(
            scene.n_points,
            scene.n_centroids,
            scene.dims,
            points_gpu.data,
            scene.initial_centroids.data,
            centroids_gpu.data,
            memory_pool);
    };

    reset();
    f();


    auto returned_centroids = std::vector<float>(scene.true_centroids.size(), 0.0f);
     CUDA_CHECK(cudaMemcpy(
        returned_centroids.data,
        centroids_gpy.data,
        scene.true_centroids.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    float squared_dist_sum = 0;

    for(int i=0;i<scene.n_centroids;i++){
        for(int j=0;j<scene.dims;j++){
            squared_dist_sum += (scene.true_centroids[i*scene.dims+j]-returned_centroids[i*scene.dims+j])
                                *(scene.true_centroids[i*scene.dims+j]-returned_centroids[i*scene.dims+j]);
        }
    }

    float average_squared_dist = squared_dist_sum/scene.n_centroids;

    double time_ms = benchmark_ms(1000.0, reset, f);

    return Results{
        average_squared_dist,
        std::move(returned_centroids),
        time_ms,
    };

}



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

    auto initial_centroids = std::vector<float>();
    for (int32_t i = 0; i < n_centroids*dims; i++) {
        float z;
        z = unif_0_1(rng);

    
        // float z = std::max(unif_0_1(rng), unif_0_1(rng));
        initial_centroids.push_back(z);
    }

    auto scene = Scene{n_points, n_centroids, dims, true_centroids, initial_centroids, features};

    return scene;

}


struct SceneTest {
    std::string name;
    Mode mode;
    Scene scene;
};

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);
    auto scenes = std::vector<SceneTest>();
    scenes.push_back(
        {"test1", Mode::BENCHMARK, gen_random(rng, 10, 65536, 10)});
    int32_t fail_count = 0;

    int32_t count = 0;
    for (auto const &scene_test : scenes) {
        auto i = count++;
        printf("\nTesting scene '%s'\n", scene_test.name.c_str());
        auto results = run_config(scene_test.mode, scene_test.scene);
    }

}


