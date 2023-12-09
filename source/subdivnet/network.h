#include <torch/torch.h>
#include <torch/script.h>

#include "mesh_ops.h"


class MeshConvBlock : public torch::nn::Module {
    using torch::nn::Module::Module;
    
    private:
        MeshConv mconv1;
        MeshConv mconv2;
        MeshBatchNorm bn1;
        MeshBatchNorm bn2;
        MeshReLU relu1;
        MeshReLU relu2;
    
    public:
        MeshConvBlock(int in_channels, 
                    int out_channels,
                    int dilation=1);

        MeshTensor forward(MeshTensor x);

};


// class MeshResIdentityBlock : public torch::nn::Module {
//     using torch::nn::Module::Module;

//     private:
//         MeshLinear conv1;
//         MeshBatchNorm bn1;
//         MeshReLU relu;
//         MeshConv conv2;
//         MeshBatchNorm bn2;
//         MeshLinear conv3;
//         MeshBatchNorm bn3;

//     public: 
//         MeshResIdentityBlock(int in_channels, 
//                         int out_channels,  
//                         int dilation=1);

//         MeshTensor forward(MeshTensor x);
// };


class MeshResConvBlock : public torch::nn::Module {
    using torch::nn::Module::Module;

    private:
        MeshLinear conv0;
        MeshBatchNorm bn0;
        MeshLinear conv1;
        MeshBatchNorm bn1;
        MeshReLU relu;
        MeshConv conv2;
        MeshBatchNorm bn2;
        MeshLinear conv3;
        MeshBatchNorm bn3;

    public: 
        MeshResConvBlock(int in_channels, 
                        int out_channels,  
                        int dilation=1);
        MeshTensor forward(MeshTensor x);
};


// class MeshBottleneck : public torch::nn::Module {
//     using torch::nn::Module::Module;

//     private:
//         MeshLinear conv1;
//         MeshBatchNorm bn1;
//         MeshReLU relu;
//         MeshConv conv2;
//         MeshBatchNorm bn2;
//         MeshLinear conv3;
//         MeshBatchNorm bn3;
//         MeshLinear downsample; // TODO: implement downsample
//         int stride;
//         int dilation;

//     public: 
//         MeshBottleneck(int inplanes, 
//                         int planes, 
//                         int stride=1, 
//                         int dilation=1, 
//                         int downsample=None);

//         MeshTensor forward(MeshTensor x);

// };

class MeshNetImpl : public torch::nn::Module {
    using torch::nn::Module::Module;
    private:
        MeshLinear fc;
        MeshReLU relu;

        torch::nn::Sequential convs{nullptr};
        
        MeshAdaptivePool global_pool;

        torch::nn::Dropout dp1{nullptr};

        torch::nn::Linear linear1{nullptr};
        torch::nn::BatchNorm1d bn{nullptr};

        torch::nn::Dropout dp2{nullptr};
        torch::nn::Linear linear2{nullptr};

        torch::nn::ReLU torch_relu{nullptr};
    public: 


        MeshNetImpl(
            int in_channels, 
            int out_channels, 
            int depth,
            std::vector<int> layer_channels,
            bool residual=false,
            std::vector<int> blocks=std::vector<int>{},
            int n_dropout=1);


        torch::Tensor forward(MeshTensor x);

};TORCH_MODULE(MeshNet);