#include "network.h"

/* MeshConvBlock */
MeshConvBlock::MeshConvBlock(int in_channels,
                            int out_channels,
                            int dilation){
    this->mconv1 = MeshConv(in_channels, out_channels, dilation=dilation, false);
    this->mconv2 = MeshConv(out_channels, out_channels, dilation=dilation, false);
    this->bn1 = MeshBatchNorm(out_channels);
    this->bn2 = MeshBatchNorm(out_channels);
    this->relu1 = MeshReLU();
    this->relu2 = MeshReLU();
}

MeshTensor MeshConvBlock::forward(MeshTensor mesh){
    mesh = this->mconv1.forward(mesh);
    mesh = this->bn1.forward(mesh);
    mesh = this->relu1.forward(mesh);
    mesh = this->mconv2.forward(mesh);
    mesh = this->bn2.forward(mesh);
    mesh = this->relu2.forward(mesh);
    return mesh;
}
/* MeshConvBlock */

/* MeshResIdentityConvBlock */
// MeshResIdentityBlock::MeshResIdentityBlock(int in_channels, 
//                         int out_channels,  
//                         int dilation){
//     this->conv1 = MeshLinear(in_channels, out_channels);
//     this->bn1 = MeshBatchNorm(out_channels);
//     this->relu = MeshReLU();
//     this->conv2 = MeshConv(out_channels, out_channels, dilation=dilation);
//     this->bn2 = MeshBatchNorm(out_channels);
//     this->conv3 = MeshLinear(out_channels, out_channels);
//     this->bn3 = MeshBatchNorm(out_channels);
// }

// MeshTensor MeshResIdentityBlock::forward(MeshTensor mesh){
//     MeshTensor identity = mesh;
//     mesh = this->conv1.forward(mesh);
//     mesh = this->bn1.forward(mesh);
//     mesh = this->relu.forward(mesh);
//     mesh = this->conv2.forward(mesh);
//     mesh = this->bn2.forward(mesh);
//     mesh = this->conv3.forward(mesh);
//     mesh = this->bn3.forward(mesh);

//     mesh.feats += identity.feats;
//     mesh = this->relu.forward(mesh);
//     return mesh;
// }
/* MeshResIdentityConvBlock */


/* MeshResConvBlock */
MeshResConvBlock::MeshResConvBlock(int in_channels, 
                        int out_channels,  
                        int dilation){
    this->conv0 = MeshLinear(in_channels, out_channels);
    this->bn0 = MeshBatchNorm(out_channels);
    this->conv1 = MeshLinear(out_channels, out_channels);
    this->bn1 = MeshBatchNorm(out_channels);
    this->relu = MeshReLU();
    this->conv2 = MeshConv(out_channels, out_channels, dilation=dilation);
    this->bn2 = MeshBatchNorm(out_channels);
    this->conv3 = MeshLinear(out_channels, out_channels);
    this->bn3 = MeshBatchNorm(out_channels);
}

MeshTensor MeshResConvBlock::forward(MeshTensor mesh){
    mesh = this->conv0.forward(mesh);
    mesh = this->bn0.forward(mesh);
    MeshTensor identity = mesh;

    mesh = this->conv1.forward(mesh);
    mesh = this->bn1.forward(mesh);
    mesh = this->relu.forward(mesh);
    mesh = this->conv2.forward(mesh);
    mesh = this->bn2.forward(mesh);
    mesh = this->conv3.forward(mesh);
    mesh = this->bn3.forward(mesh);

    mesh.feats += identity.feats;
    mesh = this->relu.forward(mesh);
    return mesh;
}
/* MeshResConvBlock */


/* MeshBottleneck */
/* MeshBottleneck */


MeshNetImpl::MeshNetImpl(
    int in_channels, 
    int out_channels, 
    int depth, 
    std::vector<int> layer_channels,
    bool residual,
    std::vector<int> blocks,
    int n_dropout) {
    this->fc = MeshLinear(in_channels, layer_channels[0]);
    this->relu = MeshReLU();

    for (int i=0; i < depth; i++){
        if (residual) {
            this->convs->push_back(MeshResConvBlock(layer_channels[i], layer_channels[i + 1]));
            for (int j=0; j<(blocks[i] - 1); j++){
                this->convs->push_back(MeshResConvBlock(layer_channels[i + 1], layer_channels[i + 1]));
            }
        } else {
            this->convs->push_back(MeshConvBlock(layer_channels[i], 
                                            layer_channels[i + 1]));
        }
        this->convs->push_back(MeshPool("max"));
        this->convs->push_back(MeshConv(layer_channels[-1], 
                                   layer_channels[-1], 
                                   false));
    }
    this->global_pool = MeshAdaptivePool("max");

    if (n_dropout >= 2){
        this->dp1 = torch::nn::Dropout(0.5);
    }
    
    this->linear1 = torch::nn::Linear(torch::nn::LinearOptions(layer_channels[-1], layer_channels[-1]).bias(false));
    this->bn = torch::nn::BatchNorm1d(layer_channels[-1]);
    this->torch_relu = torch::nn::ReLU(torch::nn::ReLUOptions(true)); // TODO: should it be set to true? 

    if (n_dropout >= 1){
        this->dp2 = torch::nn::Dropout(0.5);
    }
    this->linear2 = torch::nn::Linear(layer_channels[-1], out_channels);

}


torch::Tensor MeshNetImpl::forward(MeshTensor mesh){
    mesh = this->fc.forward(mesh);
    mesh = this->relu.forward(mesh);

    torch::Tensor mmesh = this->convs->forward(mesh);

    torch::Tensor x = this->global_pool.forward(mesh);

    // check if dp1 has been initialized
    if (this->dp1){
        x = this->dp1(x);
    }
        x = this->dp1(x);
    x = this->torch_relu(this->bn(this->linear1(x)));
    if (this->dp2){
        x = this->dp2(x);
    }
    x = this->linear2(x);
    return x;
}