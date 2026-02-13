#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class OrtInfer
{
public:
    OrtInfer(const std::string &model_path, int device_id = 0);
    std::vector<float> infer(const std::vector<float> &input, size_t batch_size);

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info;

    std::vector<const char *> input_names;
    std::vector<const char *> output_names;

    std::vector<Ort::AllocatedStringPtr> owned_input_names;
    std::vector<Ort::AllocatedStringPtr> owned_output_names;
};
