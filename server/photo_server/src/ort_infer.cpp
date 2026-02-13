#include "ort_infer.h"
#include <iostream>

OrtInfer::OrtInfer(const std::string &model_path, int device_id)
    : env(ORT_LOGGING_LEVEL_WARNING, "cifar10"), memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 🔥 CUDA Execution Provider
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = device_id;
    cuda_options.arena_extend_strategy = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.do_copy_in_default_stream = 1;

    session_options.AppendExecutionProvider_CUDA(cuda_options);

    session = Ort::Session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    owned_input_names.push_back(session.GetInputNameAllocated(0, allocator));
    owned_output_names.push_back(session.GetOutputNameAllocated(0, allocator));

    input_names.push_back(owned_input_names.back().get());
    output_names.push_back(owned_output_names.back().get());
}

std::vector<float> OrtInfer::infer(const std::vector<float> &input, size_t batch_size)
{
    // CIFAR-10: [batch_size, 3, 32, 32]
    std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), 3, 32, 32};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float *>(input.data()),
        input.size(),
        input_shape.data(),
        input_shape.size()
    );

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1);

    float *output_data =
        output_tensors[0].GetTensorMutableData<float>();

    // CIFAR-10 output: [1, 10]
    return std::vector<float>(output_data, output_data + 10 * batch_size);
}
