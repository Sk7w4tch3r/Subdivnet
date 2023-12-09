#include "mesh_ops.h"
#include <stdexcept>


MeshConv::MeshConv(int in_channels, 
                    int out_channels, 
                    int kernel_size, 
                    int dilation, 
                    int stride, 
                    bool bias){
            
    this->k_in_channels = in_channels;
    this->k_out_channels = out_channels;
    this->k_kernel_size = kernel_size;
    this->k_dilation = dilation;
    this->k_stride = stride;

    assert(this->k_kernel_size % 2 == 1);

    if (this->k_kernel_size == 1){
        assert(this->k_dilation == 1);
        this->conv1d = torch::nn::Conv1d(torch::nn::Conv1dOptions(this->k_in_channels, 
                                                                this->k_out_channels, 
                                                                this->k_kernel_size)
                                                                .bias(bias));
    } else {
        int kernel_size = 4;
        this->conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(this->k_in_channels, 
                                                                this->k_out_channels, 
                                                                (1, kernel_size))
                                                                .bias(bias));
    }

    assert(this->k_stride == 1 || this->k_stride == 2);
}

MeshTensor MeshConv::forward(MeshTensor meshTensor){
    if (this->k_in_channels != meshTensor.feats.size(1)){
        // raise error
        throw std::invalid_argument("Input channels of MeshConv must be equal to the number of features in the input MeshTensor");
    }

    if (this->k_kernel_size == 1){
        torch::Tensor feats = meshTensor.feats;
        if (this->k_stride == 2){
            int N = feats.size(0);
            int C = feats.size(1);
            int F = feats.size(2);
            meshTensor = meshTensor.inverseLoopPool("max", feats);
        }
        torch::Tensor y = this->conv1d->forward(feats);
    } else {
        meshTensor.feats = this->conv2d->forward(meshTensor.feats);
    }



    return meshTensor;
}

MeshTensor MeshPool::forward(MeshTensor meshTensor){
    return meshTensor.inverseLoopPool(this->k_op);
}

MeshTensor MeshUnpool::forward(MeshTensor meshTensor){
    return meshTensor;
}


torch::Tensor MeshAdaptivePool::forward(MeshTensor meshTensor){
    return meshTensor.feats;
}


std::vector<MeshTensor> meshConcat(std::vector<MeshTensor> meshTensors){
    return std::vector<MeshTensor>();
}


MeshTensor meshAdd(MeshTensor meshTensor1, MeshTensor meshTensor2){
    return meshTensor1;
}