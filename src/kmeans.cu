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
#include <cub/device/device_segmented_reduce.cuh>



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


struct AddOp
{
    template <typename T>
    __host__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a+b;
    }
};

AddOp add_op;


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
    if(point_idx>=n) return;
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
    float *global_dist_sums,
    float *global_point_counts
){
    int dim = threadIdx.x;
    int point_idx = threadIdx.y*points_per_thread+threadIdx.z*points_per_thread*warp_size_2
                    +blockIdx.x*points_per_thread*warp_size_2*block_size_2;


    int output_idx = point_idx/points_per_thread;


    

    

    cuda::std::unordered_map<int, float> sum_map;
    cuda::std::unordered_map<int, int> count_map;

    for(int p=point_idx;p<point_idx+points_per_thread;p++){
        if(p>=n) break;
        int label = centroid_map[p];
        if(sum_map.contains(centroid_map[p])){
            if(dim==0)
                count_map[p]++;
            sum_map[p] += points[p*d+dim];
        }
        else{
            if(dim==0)
                count_map[p]=1;
            sum_map[p] += points[p*d+dim];
        }
    }
    //__syncthreads();
    for(int i=0;i<k;i++){
        if(sum_map.contains(i)){
            if(dim===0){
                global_point_counts[((n/points_per_thread)+1)*i+output_idx] = count_map[i];
            }
            global_dist_sums[((n/points_per_thread)+1)*(i*d+dim)+output_idx] = sum_map[i];
        }
    }

    
}


__global__ void reset_centroids(
    int n,
    int k,
    int d,
    int *point_counts,
    float *centroid_sums,
    float *initial_centroids
){
    for(int c=0;c<k;c++){
        for(int dim=0;dim<d;dim++){
            initial_centroids[c*d+dim]=centroid_sums[c*d+dim]/point_counts[c];
        }
    }
}



void launch_kmeans(
    int n,
    int k,
    int d,
    float *points,
    float *initial_centroids,
    float *output_centroids,
    int *count_offsets_gpu,
    int *sum_offsets_gpu,
    int *total_counts_gpu,
    float *total_sums_gpu,
    GpuMemoryPool &memory_pool
){


    int *centroid_map = reinterpret_cast<int *>(memory_pool.alloc(n*sizeof(int)));
    int *point_counts = reinterpret_cast<int *>(memory_pool.alloc(n*sizeof(int)/points_per_thread));
    float *dist_sums = reinterpret_cast<float *>(memory_pool.alloc(n*d*sizeof(float)/points_per_thread));

    void* d_temp_storage      = nullptr;
    size_t temp_count_bytes = 0;

    

    cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage,
    temp_count_bytes,
    point_counts,
    total_counts_gpu,
    k,
    count_offsets_gpu,
    count_offsets_gpu + 1,
    add_op,
    initial_value);

    size_t temp_sum_bytes = 0;
    cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage,
    temp_sum_bytes,
    dist_sums,
    total_sums_gpu,
    k*d,
    sum_offsets_gpu,
    sum_offsets_gpu + 1,
    add_op,
    initial_value);


    uint8_t *temp_count_storage = reinterpret_cast<uint8_t *>(memory_pool.alloc(temp_count_bytes));
    uint8_t *temp_sum_storage = reinterpret_cast<uint8_t *>(memory_pool.alloc(temp_sum_bytes));


    // step 1: get clusters of points
    // step 2: get local point counts and point sums

    // step 3: reduce point counts and sums

    //https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html#_CPPv4N3cub12DeviceReduceE

    //https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedReduce.html

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;


    for(int i=0;i<100;i++){
        dim3 thread_dims_1 = dim3(warp_size, block_size);

        dim3 thread_dims_2 = dim3(d,warp_size_2, block_size_2);

        int num_blocks_2 = n/warp_size_2/block_size_2/points_per_thread+1;

        compute_clusters<<<n/warp_size/block_size+1,thread_dims_1>>>(n,k,d,points,initial_centroids,centroid_map);
        compute_centroids<<<num_blocks_2,thread_dims_2>>>(n,k,d,points,initial_centroids,centroid_map,dist_sums,point_counts);

        cub::DeviceSegmentedReduce::Reduce(
        temp_count_storage,
        temp_count_bytes,
        point_counts,
        total_counts_gpu,
        k,
        count_offsets_gpu,
        count_offsets_gpu + 1,
        add_op,
        initial_value);

        cub::DeviceSegmentedReduce::Reduce(
        temp_sum_storage,
        temp_sum_bytes,
        dist_sums,
        total_sums_gpu,
        k*d,
        sum_offsets_gpu,
        sum_offsets_gpu + 1,
        add_op,
        initial_value);

        reset_centroids<<<1,1>>>(n,k,d,total_counts_gpu,total_sums_gpu,initial_centroids);
    }

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


    int num_count_segments = scene.n_centroids;
    int num_sum_segments = scene.n_centroids * scene.dims;

    int segment_size = ((scene.n_points/kmeans.points_per_thread)+1);

    vector<int> count_offsets;
    vector<int> sum_offsets;
    for(int i=0;i<num_count_segments+1;i++){
        count_offsets.push_back(i*segment_size);
    }
    for(int i=0;i<num_sum_segments+1;i++){
        sum_offsets.push_back(i*segment_size);
    }


    vector<int> total_counts(num_count_segments);
    vector<int> total_sums(num_sum_segments);


    auto count_offsets_gpu = GpuBuf<int>(count_offsets);
    auto sum_offsets_gpu = GpuBuf<int>(sum_offsets);

    auto total_counts_gpu = GpuBuf<int>(total_counts);
    auto total_sums_gpu = GpuBuf<float>(total_sums);

    


    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(points_gpu.data, 0, scene.features.size() * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(centroids_gpu.data, 0, scene.true_centroids.size() * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(count_offsets_gpu.data, 0, (num_count_segments+1) * sizeof(int)));
        CUDA_CHECK(
            cudaMemset(sum_offsets_gpu.data, 0, (num_sum_segments+1) * sizeof(int)));
        CUDA_CHECK(
            cudaMemset(total_counts_gpu.data, 0, (num_count_segments) * sizeof(int)));
        CUDA_CHECK(
            cudaMemset(total_sums_gpu.data, 0, (num_sum_segments) * sizeof(float)));

        memory_pool.reset();
    };

    auto f = [&]() {
        kmeans::launch_kmeans(
            scene.n_points,
            scene.n_centroids,
            scene.dims,
            points_gpu.data,
            scene.initial_centroids.data,
            centroids_gpu.data,
            count_offsets_gpu.data,
            sum_offsets_gpu.data,
            total_counts_gpu.data,
            total_sums_gpu.data,
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


