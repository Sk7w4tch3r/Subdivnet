#pragma once

#include <torch/torch.h>

struct Net : torch::nn::Module {


    torch::nn::Conv2d cv1{nullptr}, cv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    Net();

    torch::Tensor forward(torch::Tensor x);

};