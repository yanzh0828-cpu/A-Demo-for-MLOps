#include "httplib.h"
#include "ort_infer.h"
#include "batch_infer_engine.h"
#include <iostream>
#include <sstream>

int main() {

    BatchInferEngine engine("../model/SimpleCNN.onnx", 0, 1);

    httplib::Server server;

    std::cout<<"[WarmUp] Inference engine is warmming!\n";
    if (!engine.WarmUp()){
        std::cout<<"[WarmUp] Inference engine warmming failed!\n Inference engine will shutdown!\n";
        return 0;
    }
    std::cout<<"[WarmUp] Inference engine warmming successful!\n";

    server.Post("/warm_http", [&](const httplib::Request& req, httplib::Response& res){
        std::cout<<"[WarmUp] HTTP engine is warmming!\n";
        res.set_content("OK", "text/plain");
        std::cout<<"[WarmUp] HTTP engine warmming successful!\n";
    });

    server.Post("/infer", [&](const httplib::Request& req, httplib::Response& res) 
    {
        // convert binary request body to tensor
        size_t num_floats = req.body.size() / sizeof(float);
        const float* data = reinterpret_cast<const float*>(req.body.data());
        std::vector<float> input(data, data + num_floats);

        //submit the inference tasks but not infer immediately
        std::future<std::vector<float>> fut = engine.submit(input);

        //waitting for the inference result of this request
        std::vector<float> output = fut.get();
        std::cout<<"output size: "<<output.size()<<std::endl;

        res.set_content(
            reinterpret_cast<const char*>(output.data()),
            output.size() * sizeof(float),
            "application/octet-stream"
        );
    });

    std::cout << "CIFAR10 server listening on http://127.0.0.1:8080\n";
    server.listen("127.0.0.1", 8080);
}