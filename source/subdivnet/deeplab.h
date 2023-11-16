#pragma once
#include <torch/torch.h>

#include "mesh_ops.h"


class MeshConvBlock : public torch::nn::Module {
    using torch::nn::Module::Module;

    public:
        MeshConv mconv1;
        MeshConv mconv2;
        MeshBatchNorm bn1;
        MeshBatchNorm bn2;
        MeshReLU relu1;
        MeshReLU relu2;

        MeshConvBlock(int in_channels, int out_channels, int dilation=1){
            this->mconv1 = MeshConv(in_channels, out_channels, 3, dilation, false);
            this->bn1 = MeshBatchNorm(out_channels);
            this->relu1 = MeshReLU();
            this->mconv2 = MeshConv(out_channels, out_channels, 3, dilation, false);
            this->bn2 = MeshBatchNorm(out_channels);
            this->relu2 = MeshReLU();
        }

        MeshTensor execute(MeshTensor x);
};


class Bottleneck : public torch::nn::Module {
    using torch::nn::Module::Module;

    public:
        MeshConvBlock conv1;
        MeshConvBlock conv2;
        MeshConvBlock conv3;

        MeshBatchNorm bn1;
        MeshBatchNorm bn2;
        MeshBatchNorm bn3;

        MeshReLU relu;
        MeshPool downsample;
        int stride;
        int dilation;

        Bottleneck(int inplanes, int planes, int stride, int dilation, MeshPool downsample);

        MeshTensor execute(MeshTensor x);
};