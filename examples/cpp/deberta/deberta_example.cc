/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/deberta/Deberta.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

using namespace fastertransformer;

template<typename T>
int debertaExample(size_t batch_size, size_t num_layers, size_t seq_len, size_t head_num, size_t size_per_head, size_t vocab_size, size_t max_relative_positions, size_t relative_position_buckets);

int main(int argc, char** argv)
{
    if (argc != 10) {
        printf("[ERROR] deberta_example <batch_size> <num_layers> <seq_len> <head_num> "
               "<size_per_head> <data_type, 0: fp32, 1: fp16, 2: bf16> <vocab_size> <max_relative_positions> <relative_position_buckets>\n");
        printf("e.g., ./bin/deberta_example 8 12 128 12 64 0 128100 512 256\n");
        return 0;
    }

    int            batch_size    = atoi(argv[1]);
    int            num_layers    = atoi(argv[2]);
    int            seq_len       = atoi(argv[3]);
    int            head_num      = atoi(argv[4]);
    int            size_per_head = atoi(argv[5]);
    FtCudaDataType data_type     = static_cast<FtCudaDataType>(atoi(argv[6]));  // 0: fp32, 1: fp16, 2: bf16
    int            vocab_size    = atoi(argv[7]);
    int            max_relative_positions    = atoi(argv[8]);
    int            relative_position_buckets = atoi(argv[9]);

    if (data_type == FP32) {
        return debertaExample<float>(batch_size, num_layers, seq_len, head_num, size_per_head, vocab_size, max_relative_positions, relative_position_buckets);
    }
#ifdef ENABLE_BF16
    else if (data_type == BF16) {
        return debertaExample<__nv_bfloat16>(batch_size, num_layers, seq_len, head_num, size_per_head, vocab_size, max_relative_positions, relative_position_buckets);
    }
#endif
    else if (data_type == FP16) {
        return debertaExample<half>(batch_size, num_layers, seq_len, head_num, size_per_head, vocab_size, max_relative_positions, relative_position_buckets);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] data_type should be fp32, fp16, or bf16 \n "));
    }
}

template<typename T>
int debertaExample(size_t batch_size, size_t num_layers, size_t seq_len, size_t head_num, size_t size_per_head, size_t vocab_size, size_t max_relative_positions, size_t relative_position_buckets)
{
    printf("[INFO] Device: %s \n", getDeviceName().c_str());

    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size   = 4 * hidden_units;

    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);

    cublasSetStream(cublas_handle, stream);
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in", "");

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex* cublas_wrapper_mutex = new std::mutex();

    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);

    if (std::is_same<T, half>::value) {
        cublas_wrapper.setFP16GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    // Set layer weight
    DebertaWeight<T> deberta_weights(hidden_units, inter_size, max_relative_positions, relative_position_buckets, vocab_size, num_layers);

    // Allocate Input & Output
    T* out_tensor;
    deviceMalloc(&out_tensor, batch_size * seq_len * hidden_units, false);

    int*         h_input_ids = new int[batch_size * seq_len];
    unsigned int seed               = 0;
    for (uint i = 0; i < batch_size * seq_len; i++) {
        h_input_ids[i] = rand_r(&seed) % vocab_size;
    }
    int* d_input_ids;
    deviceMalloc(&d_input_ids, batch_size * seq_len, false);
    cudaH2Dcpy(d_input_ids, h_input_ids, batch_size * seq_len);
    delete[] h_input_ids;

    int*         h_sequence_lengths = new int[batch_size];
    for (uint i = 0; i < batch_size; i++) {
        h_sequence_lengths[i] = seq_len;
    }
    int* d_sequence_lengths;
    deviceMalloc(&d_sequence_lengths, batch_size, false);
    cudaH2Dcpy(d_sequence_lengths, h_sequence_lengths, batch_size);
    delete[] h_sequence_lengths;

    std::vector<Tensor> input_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, seq_len}, d_input_ids},
        Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, d_sequence_lengths}};

    std::vector<Tensor> output_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, seq_len, hidden_units}, out_tensor}};

    Deberta<T> deberta = Deberta<T>(batch_size,
                              seq_len,
                              head_num,
                              size_per_head,
                              max_relative_positions,
                              relative_position_buckets,
                              inter_size,
                              num_layers,
                              1.0f,
                              stream,
                              &cublas_wrapper,
                              &allocator,
                              false,
                              false,
                              ActivationType::Gelu,
                              LayerNormType::post_layernorm);

    // warmup
    for (int i = 0; i < 10; i++) {
        deberta.forward(&output_tensors, &input_tensors, &deberta_weights);
    }

    // profile time
    const int ite = 100;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    for (int i = 0; i < ite; i++) {
        deberta.forward(&output_tensors, &input_tensors, &deberta_weights);
    }
    float total_time = cuda_timer.stop();

    printf("[INFO] batch_size %ld seq_len %ld layer %ld "
           "FT-CPP-time %.2f ms (%d iterations) \n",
           batch_size,
           seq_len,
           num_layers,
           total_time / ite,
           ite);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    cudaFree(d_input_ids);
    deviceFree(d_sequence_lengths);
    cudaFree(out_tensor);

    return 0;
}
