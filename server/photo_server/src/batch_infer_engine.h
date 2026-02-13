#pragma once
#include "ort_infer.h"
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include <cuda_runtime.h>

class BatchInferEngine {
    public:
        BatchInferEngine(const std::string &model_path, int device_id, size_t batch_size);
        ~BatchInferEngine();

        //append the single request to the queue and return a inference result by using future
        std::future<std::vector<float>> submit(const std::vector<float> &input);

        //its using to WarmUp inference engine
        bool WarmUp();

    private:
        void batch_worker(); //back thread, handle the queue and batch inference; check the queue when attach the size of batch_size, infering together

        OrtInfer infer;
        size_t batch_size;

        std::queue<std::pair<std::vector<float>, std::promise<std::vector<float>>>> request_queue; //to store the inference request
        std::mutex queue_mutex;
        std::condition_variable queue_cv;

        bool stop_flag;  //to control the thread out
        std::thread worker_thread;
};