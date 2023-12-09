#pragma once
#include <string>
#include <givde/lib/dggs/multiatlas.h>

#include <torch/torch.h>
#include <vector>

// #include <iostream>
#include <cassert>

#include "mesh_tensor.h"


using Index = givde::multiatlas::Index;

class MeshConv : public torch::nn::Module {
    // inherit the constructor of Module
    using torch::nn::Module::Module;
    
    public:
        int k_in_channels;
        int k_out_channels;
        int k_kernel_size;
        int k_dilation;
        int k_stride;

        torch::nn::Conv1d conv1d = nullptr;
        torch::nn::Conv2d conv2d = nullptr;
        
        MeshConv(int in_channels, 
                    int out_channels, 
                    int kernel_size=3, 
                    int dilation=1, 
                    int stride=1, 
                    bool bias = true);

        MeshTensor forward(MeshTensor meshTensor);

};


class MeshPool : public torch::nn::Module {
    // inherit the constructor of Module
    using torch::nn::Module::Module;
    
    public:
        std::string k_op;

        MeshPool(std::string op){
            this->k_op = op;
        }

        MeshTensor forward(MeshTensor meshTensor);

};


class MeshUnpool : public torch::nn::Module {
    using torch::nn::Module::Module;

    public:
        int k_mode;

        MeshUnpool(int mode){
            this->k_mode = mode;
        }

        MeshTensor forward(MeshTensor meshTensor);
};


class MeshAdaptivePool : public torch::nn::Module {
    using torch::nn::Module::Module;

    public:
        std::string k_mode;

        MeshAdaptivePool(std::string mode){
            this->k_mode = mode;
        }

        torch::Tensor forward(MeshTensor meshTensor);
};


class MeshBatchNorm : public torch::nn::Module {
    using torch::nn::Module::Module;

    private:
        float k_eps = 1e-5;
        float k_momentum = 0.1;

    public:
        torch::nn::BatchNorm2d bn = nullptr;


        MeshBatchNorm(int num_features){
            this->bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features).eps(this->k_eps).momentum(this->k_momentum));
        }

        MeshTensor forward(MeshTensor meshTensor){
            torch::Tensor feats = this->bn->forward(meshTensor.feats);
            return meshTensor.updated(feats);
        };
};


class MeshReLU : public torch::nn::Module {
    using torch::nn::Module::Module;

    public:
        torch::nn::ReLU relu;

        MeshReLU(){
            this->relu = torch::nn::ReLU();
        }

        MeshTensor forward(MeshTensor meshTensor){
            torch::Tensor feats = this->relu->forward(meshTensor.feats);
            return meshTensor.updated(feats);
        }
};


class MeshDropout : public torch::nn::Module {
    using torch::nn::Module::Module;

    public:
        torch::nn::Dropout dropout = nullptr;

        MeshDropout(float p){
            this->dropout = torch::nn::Dropout(torch::nn::DropoutOptions(p));
        }
        
        MeshTensor forward(MeshTensor meshTensor){
            torch::Tensor feats = this->dropout->forward(meshTensor.feats);
            return meshTensor.updated(feats);
        }
};


class MeshLinear : public torch::nn::Module {
    using torch::nn::Module::Module;

    public:
        torch::nn::Conv1d conv1d = nullptr;

        MeshLinear(int in_channels, int out_channels, bool bias = true){
            this->conv1d = torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, 1).bias(bias));
        }

        MeshTensor forward(MeshTensor meshTensor){
            torch::Tensor feats = this->conv1d->forward(meshTensor.feats);
            return meshTensor.updated(feats);
        }
};

