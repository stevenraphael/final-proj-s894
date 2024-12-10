#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

//#include <cuda/std/unordered_map>
#include <cub/device/device_segmented_reduce.cuh>


////////////////////////////////////////////////////////////////////////////////
// Utility Functions

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

class GpuMemoryPool {
  public:
    GpuMemoryPool() = default;

    ~GpuMemoryPool();

    GpuMemoryPool(GpuMemoryPool const &) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool const &) = delete;
    GpuMemoryPool(GpuMemoryPool &&) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool &&) = delete;

    void *alloc(size_t size);
    void reset();

  private:
    std::vector<void *> allocations_;
    std::vector<size_t> capacities_;
    size_t next_idx_ = 0;
};



enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename T> struct GpuBuf {
    T *data;

    explicit GpuBuf(size_t n) { CUDA_CHECK(cudaMalloc(&data, n * sizeof(T))); }

    explicit GpuBuf(std::vector<T> const &host_data) {
        CUDA_CHECK(cudaMalloc(&data, host_data.size() * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(
            data,
            host_data.data(),
            host_data.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    ~GpuBuf() { CUDA_CHECK(cudaFree(data)); }
};

const int micro_i = 16;
const int micro_j = 8;
const int micro_k = 8;

const int CENTROIDBOUND = 64;
const int num_rows = 4;

const int num_cols = 4;

const int ROWBOUND = micro_i*num_rows;


struct shmemstruct{
    float buf1[num_rows*micro_i*CENTROIDBOUND];
};

__global__ void matmul(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    uint32_t *centroid_map, /* pointer to GPU memory */
    float *asquared,
    float *bsquared
){
    extern __shared__ shmemstruct my_shared_memory[];
    float* buf1 = my_shared_memory->buf1;

    int idx = threadIdx.x;
    int i_oo = blockIdx.x * num_rows*micro_i;


    int idx2 = idx + micro_j*num_cols*threadIdx.y;


    //int j_oo = 0;

    float best_dists[4];
    int best_cents[4] = {0,0,0,0};

    for(int cent_offset=0;cent_offset<size_j;cent_offset+=CENTROIDBOUND){
        int upper_j1 = min(size_j, CENTROIDBOUND+cent_offset);


        /*for(int i=idx+i_oo;i<i_oo+micro_i*num_rows;i+=32){
            if(i>=size_i) break;
            for(int j=0;j<CENTROIDBOUND;j++){
                if(j+cent_offset>=size_j)
                    break;
                buf1[j*ROWBOUND+(i-i_oo)] = 0;//asquared[i]+bsquared[j+cent_offset];


            }        
        }*/
        __syncthreads();
        int j_oo=cent_offset+threadIdx.y*num_cols*micro_j;
        //for(int j_oo=cent_offset;j_oo<cent_offset+CENTROIDBOUND;j_oo+=num_cols*micro_j){
            uint32_t b_micro[num_cols][2];
            uint32_t c_micro[num_rows][num_cols][4];
            for(int x=0;x<num_rows;x++){
                for(int y=0;y<num_cols;y++){
                    for(int z=0;z<4;z++){
                        c_micro[x][y][z] = __float_as_uint(0.0);
                    }
                }
            }
            int upper_cols = min(num_cols, (size_j-j_oo)/micro_j+1);
            for(int k_outer=0; k_outer<size_k;k_outer+=micro_k){

                
                for(int j_outer=0;j_outer<upper_cols;j_outer++){
                    bool oob_j = false;//((idx/4+j_outer*micro_j+j_oo) >= size_j);

                    bool oob_k1 = false;//(oob_j||((idx%4+k_outer) > size_k));
                    bool oob_k2 = false;//(oob_j||((4+idx%4+k_outer) > size_k));
                    b_micro[j_outer][0] = __float_as_uint(oob_k1 ? 0.0 : *(b+size_k*(idx/4+j_outer*micro_j+j_oo)+idx%4+k_outer));
                    b_micro[j_outer][1] = __float_as_uint(oob_k2 ? 0.0 : *(b+size_k*(idx/4+j_outer*micro_j+j_oo)+4+idx%4+k_outer));
                }
                for(int i_outer=0;i_outer<num_rows;i_outer++){
                    
                    bool oob_i1 = false;//((i_outer*micro_i+idx/4+i_oo) > size_i);
                    bool oob_i2 = false;//((i_outer*micro_i+idx/4+i_oo+8) > size_i);
                    bool oob_k1 = false;//(((idx%4+k_outer) > size_k));
                    bool oob_k2 = false;//(((4+idx%4+k_outer) > size_k));
                    uint32_t a1 = __float_as_uint((oob_i1 || oob_k1) ? 0.0 : __ldg(a+idx%4+size_k*(i_outer*micro_i+idx/4+i_oo)+k_outer));
                    uint32_t a2 = __float_as_uint((oob_i2 || oob_k1) ? 0.0 : __ldg(a+8*size_k+idx%4+size_k*(i_outer*micro_i+idx/4+i_oo)+k_outer));
                    uint32_t a3 = __float_as_uint((oob_i1 || oob_k2) ? 0.0 : __ldg(a+4+idx%4+size_k*(i_outer*micro_i+idx/4+i_oo)+k_outer));
                    uint32_t a4 = __float_as_uint((oob_i2 || oob_k2) ? 0.0 : __ldg(a+8*size_k+4+idx%4+size_k*(i_outer*micro_i+idx/4+i_oo)+k_outer));
                    for(int j_outer=0;j_outer<upper_cols;j_outer++){
                        int idx = threadIdx.x;
                        uint32_t d1 = c_micro[i_outer][j_outer][0];
                        uint32_t d2 = c_micro[i_outer][j_outer][1];
                        uint32_t d3 = c_micro[i_outer][j_outer][2];
                        uint32_t d4 = c_micro[i_outer][j_outer][3];


                        uint32_t b1 = b_micro[j_outer][0];
                        uint32_t b2 = b_micro[j_outer][1];
                        asm(
                            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3},     {%4, %5, %6, %7},  {%8, %9},  {%10, %11, %12, %13};"
                            : "+r"(d1), "+r"(d2), "+r"(d3), "+r"(d4)
                            : "r"(a1), "r"(a2), "r"(a3), "r"(a4),
                            "r"(b1), "r"(b2),
                            "r"(d1), "r"(d2), "r"(d3), "r"(d4)
                        );
                        c_micro[i_outer][j_outer][0] = d1;
                        c_micro[i_outer][j_outer][1] = d2;
                        c_micro[i_outer][j_outer][2] = d3;
                        c_micro[i_outer][j_outer][3] = d4;
                    }
                }
            }
            for(int i_outer=0;i_outer<num_rows;i_outer++){
                for(int j_outer=0;j_outer<upper_cols;j_outer++){
                    buf1[(2*(idx%4)+(j_oo-cent_offset)+j_outer*micro_j)*ROWBOUND+(idx/4+i_outer*micro_i)] = -2*__uint_as_float(c_micro[i_outer][j_outer][0]);
                    buf1[(1+2*(idx%4)+(j_oo-cent_offset)+j_outer*micro_j)*ROWBOUND+(idx/4+i_outer*micro_i)] = -2*__uint_as_float(c_micro[i_outer][j_outer][1]);
                    buf1[(2*(idx%4)+(j_oo-cent_offset)+j_outer*micro_j)*ROWBOUND+(8+idx/4+i_outer*micro_i)] = -2*__uint_as_float(c_micro[i_outer][j_outer][2]);
                    buf1[(1+2*(idx%4)+(j_oo-cent_offset)+j_outer*micro_j)*ROWBOUND+(8+idx/4+i_outer*micro_i)] = -2*__uint_as_float(c_micro[i_outer][j_outer][3]);
                }
            }
        //}
        __syncthreads();

        int upper_j = min(cent_offset+CENTROIDBOUND, size_j);

        for(int i=idx2+i_oo;i<i_oo+micro_i*num_rows;i+=64){
            if(i>=size_i) break;
            //float best_dist = buf1[(i-i_oo)*CENTROIDBOUND];
            //int best_cent = 0;
            int inner = (i-i_oo)/32;
            float new_dist = buf1[(i-i_oo)]+asquared[i]+bsquared[cent_offset];
            if(cent_offset==0||new_dist<best_dists[inner]){
                best_dists[inner] = new_dist;
                best_cents[inner] = cent_offset;
            }
            for(int j=cent_offset+1;j<upper_j;j++){
                float new_dist = buf1[(i-i_oo)+ROWBOUND*(j-cent_offset)]+asquared[i]+bsquared[j];
                if(new_dist<best_dists[inner]){
                    best_dists[inner] = new_dist;
                    best_cents[inner] = j;
                }
            }
            //if(upper_j>=size_j)
            centroid_map[i] = best_cents[inner];
        }
        __syncthreads();
    }

}


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
    __host__ __device__ __forceinline__
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

__global__ void compute_clusters(
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
        float next_dist = 0;
        for(int idx=0;idx<d;idx++){
            float point_coord = points[point_idx*d+idx];
            float centroid_coord = centroids[i*d+idx];
            next_dist += (point_coord-centroid_coord)*(point_coord-centroid_coord);
        }
        if(next_dist<curr_dist){
            curr_dist = next_dist;
            best_centroid = i;
        }
    }
    centroid_map[point_idx] = best_centroid;
}


const int points_per_thread = 4000;


const int warp_size_2 = 32;

const int block_size_2 = 4;


const int MAX_CENTROIDS = 128;



__global__ void compute_counts(
    int n,
    int k,
    int d,
    float *points,
    float *centroids,
    uint32_t *centroid_map,
    float *global_dist_sums,
    int *global_point_counts
){
    int point_idx = threadIdx.x*points_per_thread+threadIdx.y*points_per_thread*warp_size_2
                    +blockIdx.x*points_per_thread*warp_size_2*block_size_2;

    int min_centroid = MAX_CENTROIDS*blockIdx.y;

    int output_idx = point_idx/points_per_thread;

    if(point_idx>=n) return;


    

    
    int count_map[MAX_CENTROIDS];

    for(int i=0;i<MAX_CENTROIDS;i++){
        count_map[i]=0;
    }
    //cuda::std::unordered_map<int, float> sum_map;
    //cuda::std::unordered_map<int, int> count_map;
    

    for(int p=point_idx;p<point_idx+points_per_thread;p++){
        if(p>=n) break;
        if(min_centroid<=centroid_map[p]&&centroid_map[p]<min_centroid+MAX_CENTROIDS)
            count_map[centroid_map[p]-min_centroid]+=1;
    }
    int upper = min(min_centroid+MAX_CENTROIDS, k);
    for(int i=min_centroid;i<upper;i++){

        global_point_counts[((n/points_per_thread)+1)*i+output_idx] = count_map[i-min_centroid];
    }
}

__global__ void compute_centroids(
    int n,
    int k,
    int d,
    float *points,
    float *centroids,
    uint32_t *centroid_map,
    float *global_dist_sums,
    int *global_point_counts
){
    
    int dim = threadIdx.x+threadIdx.z*32;
    int point_idx = threadIdx.y*points_per_thread+blockIdx.x*points_per_thread*warp_size_2
                    +blockIdx.y*points_per_thread*warp_size_2*block_size_2;

    int output_idx = point_idx/points_per_thread;

    int min_centroid = MAX_CENTROIDS*blockIdx.z;


    if(dim>=d) return;

    if(point_idx>=n) return;


    

    
    float sum_map[MAX_CENTROIDS];

    for(int i=0;i<MAX_CENTROIDS;i++){
        sum_map[i]=0.0;
    }
    //cuda::std::unordered_map<int, float> sum_map;
    //cuda::std::unordered_map<int, int> count_map;
    

    for(int p=point_idx;p<point_idx+points_per_thread;p++){
        if(p>=n) break;
        //int label = centroid_map[p];
        /*if(sum_map.contains(centroid_map[p])){
            if(dim==0)
                count_map[p]++;
            sum_map[p] += points[p*d+dim];
        }*/
        //else{
        if(min_centroid<=centroid_map[p]&&centroid_map[p]<min_centroid+MAX_CENTROIDS)
            sum_map[centroid_map[p]-min_centroid] += points[p*d+dim];
        //}
    }
    //__syncthreads();

    //if(blockIdx.x!=0) return;

    int upper = min(min_centroid+MAX_CENTROIDS, k);
    
    for(int i=min_centroid;i<upper;i++){
        //if(sum_map.contains(i)){
            
            //if(point_idx==0&&dim==0){
            global_dist_sums[((n/points_per_thread)+1)*(i*d+dim)+output_idx] = sum_map[i-min_centroid];
            //global_dist_sums[0]=500.0;
            //}
        //}
    }

    
}


__global__ void reset_centroids(
    int n,
    int k,
    int d,
    int *point_counts,
    float *centroid_sums,
    float *initial_centroids,
    int *sum_offsets_gpu
){
    //printf("%D %d\n",k,d);
    for(int c=0;c<k;c++){
        for(int dim=0;dim<d;dim++){
                //printf("%f %d\n",centroid_sums[c*d+dim], point_counts[c]);
            if(point_counts[c]>0)
                initial_centroids[c*d+dim]=centroid_sums[c*d+dim]/point_counts[c];
        }
    }
    
}


const int squared_group_size = 32;

__global__ void compute_squared_dists(
    int dim1,
    int dim2,
    float *in,
    float *out
){
    int offset = threadIdx.x + blockIdx.x*squared_group_size;
    if(offset<dim1){
        float x = 0;
        for(int i=0;i<dim2;i++){
            x += in[offset*dim2+i] * in[offset*dim2+i];
        }
        out[offset] = x;
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

    uint32_t shmem_size_bytes = CENTROIDBOUND*micro_i*num_rows*sizeof(float);
        /*CUDA_CHECK(cudaFuncSetAttribute(
            matmul,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size_bytes));*/


    uint32_t *centroid_map = reinterpret_cast<uint32_t *>(memory_pool.alloc(n*sizeof(uint32_t)));
    int *point_counts = reinterpret_cast<int *>(memory_pool.alloc((n/points_per_thread+1)*k*sizeof(int)));
    float *dist_sums = reinterpret_cast<float *>(memory_pool.alloc((n/points_per_thread+1)*k*d*sizeof(float)));


    float *squared_points = reinterpret_cast<float *>(memory_pool.alloc(n*sizeof(float)));
    float *squared_centroids = reinterpret_cast<float *>(memory_pool.alloc(k*sizeof(float)));

    void* d_temp_storage      = nullptr;
    size_t temp_count_bytes = 0;

    int initial_value = 0;


    compute_squared_dists<<<n/32+1, 32>>>(n,d,points,squared_points);

    

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

    //void* d_temp_storage      = nullptr;
    //size_t temp_storage_bytes = 0;

    dim3 griddims0 = dim3(n/(num_rows*micro_i)+1);


    for(int i=0;i<1;i++){
        compute_squared_dists<<<k/32+1, 32>>>(k,d,initial_centroids,squared_centroids);
        dim3 thread_dims_1 = dim3(32,2);

        dim3 thread_dims_2 = dim3(32, warp_size_2, 1+(d-1)/32);
        dim3 block_dims_2 = dim3(block_size_2, n/warp_size_2/block_size_2/points_per_thread+1, 1+(k-1)/MAX_CENTROIDS);

        dim3 num_blocks_3 = dim3(n/warp_size_2/block_size_2/points_per_thread+1, 1+(k-1)/MAX_CENTROIDS);

        dim3 thread_dims_3 = dim3(warp_size_2, block_size_2);


        matmul<<<griddims0, thread_dims_1, shmem_size_bytes>>>(n,k,d, points, initial_centroids, centroid_map, squared_points, squared_centroids);
        //compute_clusters<<<n/warp_size/block_size+1,thread_dims_1>>>(n,k,d,points,initial_centroids,centroid_map);
        //compute_centroids<<<num_blocks_2,thread_dims_2>>>(n,k,d,points,initial_centroids,centroid_map,dist_sums,point_counts);
        compute_centroids<<<block_dims_2,thread_dims_2>>>(n,k,d,points,initial_centroids,centroid_map,dist_sums,point_counts);
        compute_counts<<<num_blocks_3,thread_dims_3>>>(n,k,d,points,initial_centroids,centroid_map,dist_sums,point_counts);
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

        //std::cout<<sum_offsets_gpu[0];//<<sum_offsets_gpu[1];

        reset_centroids<<<1,1>>>(n,k,d,total_counts_gpu,total_sums_gpu,initial_centroids, sum_offsets_gpu);
        //reset_centroids<<<1,1>>>(n,k,d,point_counts,dist_sums,initial_centroids);
    }

}


}




GpuMemoryPool::~GpuMemoryPool() {
    for (auto ptr : allocations_) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void *GpuMemoryPool::alloc(size_t size) {
    if (next_idx_ < allocations_.size()) {
        auto idx = next_idx_++;
        if (size > capacities_.at(idx)) {
            CUDA_CHECK(cudaFree(allocations_.at(idx)));
            CUDA_CHECK(cudaMalloc(&allocations_.at(idx), size));
            CUDA_CHECK(cudaMemset(allocations_.at(idx), 0, size));
            capacities_.at(idx) = size;
        }
        return allocations_.at(idx);
    } else {
        void *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        CUDA_CHECK(cudaMemset(ptr, 0, size));
        allocations_.push_back(ptr);
        capacities_.push_back(size);
        next_idx_++;
        return ptr;
    }
}

void GpuMemoryPool::reset() {
    next_idx_ = 0;
    for (int32_t i = 0; i < allocations_.size(); i++) {
        CUDA_CHECK(cudaMemset(allocations_.at(i), 0, capacities_.at(i)));
    }
}

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}



struct Results {
    float average_squared_dist;
    std::vector<float> centroids;
    double time_ms;
};


Results run_config(Mode mode, Scene &scene) {
    auto points_gpu = GpuBuf<float>(scene.features);
    auto centroids_gpu = GpuBuf<float>(scene.initial_centroids);
    //auto initial_centroids_gpu = GpuBuf<float>(scene.initial_centroids);
    auto memory_pool = GpuMemoryPool();


    int num_count_segments = scene.n_centroids;
    int num_sum_segments = scene.n_centroids * scene.dims;

    int segment_size = ((scene.n_points/kmeans::points_per_thread)+1);

    std::vector<int> count_offsets;
    std::vector<int> sum_offsets;
    for(int i=0;i<num_count_segments+1;i++){
        count_offsets.push_back(i*segment_size);
    }
    for(int i=0;i<num_sum_segments+1;i++){
        sum_offsets.push_back(i*segment_size);
    }


    std::vector<int> total_counts(num_count_segments);
    std::vector<float> total_sums(num_sum_segments);

    //printf("%d %d",sum_offsets[0],sum_offsets[1]);


    auto count_offsets_gpu = GpuBuf<int>(count_offsets);
    auto sum_offsets_gpu = GpuBuf<int>(sum_offsets);

    auto total_counts_gpu = GpuBuf<int>(total_counts);
    auto total_sums_gpu = GpuBuf<float>(total_sums);

    


    auto reset = [&]() {
        /*CUDA_CHECK(
            cudaMemset(points_gpu.data, 0, scene.features.size() * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(centroids_gpu.data, 0, scene.true_centroids.size() * sizeof(float)));*/
        /*CUDA_CHECK(
            cudaMemset(count_offsets_gpu.data, 0, (num_count_segments+1) * sizeof(int)));
        CUDA_CHECK(
            cudaMemset(sum_offsets_gpu.data, 0, (num_sum_segments+1) * sizeof(int)));*/
        CUDA_CHECK(
            cudaMemset(total_counts_gpu.data, 0, (num_count_segments) * sizeof(int)));
        CUDA_CHECK(
            cudaMemset(total_sums_gpu.data, 0, (num_sum_segments) * sizeof(float)));

        memory_pool.reset();
        CUDA_CHECK(cudaMemcpy(
        centroids_gpu.data,
        scene.initial_centroids.data(),
        scene.initial_centroids.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    };

    auto f = [&]() {
        kmeans::launch_kmeans(
            scene.n_points,
            scene.n_centroids,
            scene.dims,
            points_gpu.data,
            centroids_gpu.data,
            centroids_gpu.data,
            count_offsets_gpu.data,
            sum_offsets_gpu.data,
            total_counts_gpu.data,
            total_sums_gpu.data,
            memory_pool);
    };

    reset();
    f();


    auto returned_centroids = std::vector<float>(scene.initial_centroids.size(), 0.0f);
     CUDA_CHECK(cudaMemcpy(
        returned_centroids.data(),
        centroids_gpu.data,
        scene.n_centroids * scene.dims * sizeof(float),
        cudaMemcpyDeviceToHost));

    float squared_dist_sum = 0;
    /*for(int i=0;i<scene.true_centroids.size();i++){
        printf("%f ", returned_centroids[i]);
        printf("%f \n", scene.true_centroids[i]);
    }*/

    for(int i=0;i<scene.n_centroids;i++){
        for(int j=0;j<scene.dims;j++){
            squared_dist_sum += (scene.true_centroids[i*scene.dims+j]-returned_centroids[i*scene.dims+j])
                                * (scene.true_centroids[i*scene.dims+j]-returned_centroids[i*scene.dims+j]);
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


template <typename Rng>
Scene gen_random(Rng &rng, int32_t dims, int32_t n_points, int32_t n_centroids){
    auto unif_100 = std::uniform_real_distribution<float>(-100.0f, 100.0f);
    auto unif_0_1 = std::uniform_real_distribution<float>(0.0f, 1.0f);
    auto true_centroids = std::vector<float>();

    const float stddev = 10.0;
    auto initial_centroids = std::vector<float>();
    auto normal = std::normal_distribution<double>(0.0, stddev);
    for (int32_t i = 0; i < (n_centroids+8)*dims; i++) {
        float z;
        z = unif_100(rng);

    
        // float z = std::max(unif_0_1(rng), unif_0_1(rng));
        true_centroids.push_back(z);
        initial_centroids.push_back(z+normal(rng));
    }

    

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
    for(int i=0;i<16*dims;i++){
        features.push_back(0.0);
    }

    
    for (int32_t i = 0; i < n_centroids*dims; i++) {
        float z;
        z = unif_100(rng);

    
        // float z = std::max(unif_0_1(rng), unif_0_1(rng));
        //initial_centroids.push_back(z);
    }

    auto scene = Scene{dims, n_points, n_centroids, true_centroids, initial_centroids, features};

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
        {"test1", Mode::BENCHMARK, gen_random(rng, 32, 2000000, 1024)});
    int32_t fail_count = 0;

    int32_t count = 0;
    for (auto &scene_test : scenes) {
        auto i = count++;
        printf("\nTesting scene '%s'\n", scene_test.name.c_str());
        auto results = run_config(scene_test.mode, scene_test.scene);
        printf("  Error: %f \n", results.average_squared_dist);
        printf("  Time: %f \n", results.time_ms);
    }

}


