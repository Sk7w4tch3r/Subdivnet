#include "conv.h"

Net::Net() {
    cv1 = register_module("cv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(1)));
    cv2 = register_module("cv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)));

    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
}

// Implement the Net's algorithm.
torch::Tensor Net::forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return x;
}