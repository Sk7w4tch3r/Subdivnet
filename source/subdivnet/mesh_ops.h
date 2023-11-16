#pragma once
#include <string>

#include <torch/torch.h>
#include <vector>

#include "mesh_tensor.h"

class MeshConv : public torch::nn::Module {
    // inherit the constructor of Module
    using torch::nn::Module::Module;
    
    public:
        int in_channels;
        int out_channels;
        int kernel_size;
        int dilation;
        int stride;

        // assert kernel_size % 2 == 1
        // assert(kernel_size % 2 == 1);
        
        MeshConv(int in_channels, int out_channels, int kernel_size, int dilation, int stride, bool bias = true){
            this->in_channels = in_channels;
            this->out_channels = out_channels;
            this->kernel_size = kernel_size;
            this->dilation = dilation;
            this->stride = stride;
        }

        MeshTensor execute(MeshTensor meshTensor);

};


class MeshPool : public torch::nn::Module {
    // inherit the constructor of Module
    using torch::nn::Module::Module;
    
    public:
        std::string op;

        MeshPool(std::string op){
            this->op = op;
        }

        MeshTensor execute(MeshTensor meshTensor);

};


class MeshUnpool : public torch::nn::Module {
    using torch::nn::Module::Module;

    public:
        int mode;

        MeshUnpool(int mode){
            this->mode = mode;
        }

        MeshTensor execute(MeshTensor meshTensor);
};


class MeshAdaptivePool : public torch::nn::Module {
    using torch::nn::Module::Module;

    public:
        std::string mode;

        MeshAdaptivePool(std::string mode){
            this->mode = mode;
        }

        MeshTensor execute(MeshTensor meshTensor);
};


class MeshBatchNorm : public torch::nn::Module {
    using torch::nn::Module::Module;

    private:
        float eps = 1e-5;
        float momentum = 0.1;

    public:
        // torch::Tensor bn;


        MeshBatchNorm(int num_features){
            // this->bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features).eps(eps).momentum(momentum));
        }

        MeshTensor execute(MeshTensor meshTensor);
};


class MeshReLU : public torch::nn::Module {
    using torch::nn::Module::Module;

    public:
        // torch::Tensor relu;

        MeshReLU(){
            // this->relu = torch::nn::ReLU();
        }

        MeshTensor execute(MeshTensor meshTensor);
};


class MeshDropout : public torch::nn::Module {};


class MeshLinear : public torch::nn::Module {};

