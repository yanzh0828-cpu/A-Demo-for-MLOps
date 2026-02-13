#include "batch_infer_engine.h"
#include <iostream>

BatchInferEngine::BatchInferEngine(const std::string &model_path, int device_id, size_t batch_size_)
    : infer(model_path, device_id), batch_size(batch_size_), stop_flag(false)
    {
        worker_thread = std::thread(&BatchInferEngine::batch_worker, this);
    }

BatchInferEngine::~BatchInferEngine() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        stop_flag = true;
    }
    queue_cv.notify_all();
    if (worker_thread.joinable())
        worker_thread.join();
}

std::future<std::vector<float>> BatchInferEngine::submit(const std::vector<float> &input) {
    std::promise<std::vector<float>> promise;
    auto fut = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        request_queue.push({input, std::move(promise)});
    }

    queue_cv.notify_one();
    return fut;
}

void BatchInferEngine::batch_worker() {
    while (true) {
        std::vector<std::pair<std::vector<float>, std::promise<std::vector<float>>>> batch;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this] {
                return stop_flag || request_queue.size() >= batch_size;
            });

            if (stop_flag && request_queue.empty()) break;

            // attech the batch_size or the queue is non-empty
            size_t n = std::min(batch_size, request_queue.size());
            for (size_t i = 0; i < n; ++i) {
                batch.push_back(std::move(request_queue.front()));
                request_queue.pop();
            }
        }

        // create batch input vector
        // assume every input have the same shape
        size_t single_input_size = batch[0].first.size();
        std::vector<float> batch_input(batch.size() * single_input_size);

        for (size_t i = 0; i < batch.size(); ++i) {
            std::copy(batch[i].first.begin(), batch[i].first.end(),
                      batch_input.begin() + i * single_input_size);
        }

        // if the onnx model support batch input, needing change the batch_input to [B, C, H, W] shape
        std::vector<float> batch_output = infer.infer(batch_input, batch_size); 

        // split the result to different request
        for (size_t i = 0; i < batch.size(); ++i) {
            // assume every output has fixed size, like 10
            std::vector<float> single_output(batch_output.begin() + i * 10, batch_output.begin() + (i + 1) * 10);
            batch[i].second.set_value(single_output); // 设置 promise
        }
    }
}

bool BatchInferEngine::WarmUp()
{
    try{    

        std::vector<float> dummy_input(batch_size * 3 * 32 * 32, 0.0f);

        infer.infer(dummy_input, batch_size);

        cudaDeviceSynchronize();

        return true;

    } catch(const std::exception& e) {

        std::cerr<<"[WarmUp] Fail to warm inference engine "<<e.what()<<std::endl;
    }

    return false;
}