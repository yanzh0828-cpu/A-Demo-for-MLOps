#include "httplib.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

std::vector<float> image_to_tensor(const cv::Mat& img, int target_h, int target_w)
{
    cv::Mat resized, float_img;

    cv::resize(img, resized, cv::Size(target_w, target_h));
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    std::vector<float> tensor;
    tensor.reserve(3 * target_h * target_w);

    for (int c = 0; c < 3; ++c)
    {
        tensor.insert(tensor.end(), (float*)channels[c].datastart, (float*)channels[c].dataend);
    }

    return tensor;
}

bool call_infer_uds(const std::vector<float>& tensor, std::vector<float>& output)
{
    constexpr int MAX_RETRY = 3;

    for (int attempt = 0; attempt < MAX_RETRY; ++attempt) {
        httplib::Client client("localhost");
        client.set_unix_socket_path("/run/infer/infer.sock");
        client.set_connection_timeout(0, 200000); // 200ms
        client.set_read_timeout(1, 0);

        auto res = client.Post(
            "/infer",
            reinterpret_cast<const char*>(tensor.data()),
            tensor.size() * sizeof(float),
            "application/octet-stream"
        );

        if (res && res->status == 200) {
            size_t n = res->body.size() / sizeof(float);
            const float* data =
                reinterpret_cast<const float*>(res->body.data());
            output.assign(data, data + n);
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    return false;
}

int main()
{
    httplib::Server server;

    std::vector<std::string> result = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

    server.Post("/predict", [&result](const httplib::Request& req, httplib::Response& res) {

        if (!req.form.has_file("image")) {
            res.status = 400;
            res.set_content("No image uploaded", "text/plain");
            std::cout<<"Can't find image\n";
            return;
        }
        
        std::cout<<"Receive request in\n";

        //
        const auto& file = req.form.get_file("image");
        std::vector<uchar> img_buf(file.content.begin(), file.content.end());
        cv::Mat img = cv::imdecode(img_buf, cv::IMREAD_COLOR);

        if (img.empty()) {
            res.status = 400;
            res.set_content("Invalid image", "text/plain");
            return;
        }

        //
        auto tensor = image_to_tensor(img, 32, 32);
        std::cout<<"convert to tensor\n";

        //
        std::cout<<"image ready to infer\n";
        std::vector<float> output;
        if (!call_infer_uds(tensor, output)) {
            res.status = 500;
            res.set_content("Inference failed", "text/plain");
            return;
        }

        if (output.size() != result.size()) {
            std::cout<<output.size()<<" vs "<<result.size()<<std::endl;
            res.status = 500;
            res.set_content("Invalid inference output size", "text/plain");
            return;
        }

        auto max_it = std::max_element(output.begin(), output.end());
        int it = std::distance(output.begin(), max_it);


        res.set_content(result[it], "text/plain");
    });

    std::cout << "Main server listening on port 12345\n";
    server.listen("127.0.0.1", 12345);

    return 0;
}