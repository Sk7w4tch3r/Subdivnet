#include "deeplab.h"


MeshTensor MeshConvBlock::execute(MeshTensor x) {
    x = this->mconv1.execute(x);
    x = this->bn1.execute(x);
    x = this->relu1.execute(x);
    x = this->mconv2.execute(x);
    x = this->bn2.execute(x);
    x = this->relu2.execute(x);
    return x;
}


MeshTensor Bottleneck::execute(MeshTensor x){
    MeshTensor residual = x;
    x = this->conv1.execute(x);
    x = this->bn1.execute(x);
    x = this->relu.execute(x);
    x = this->conv2.execute(x);
    x = this->bn2.execute(x);
    x = this->relu.execute(x);
    x = this->conv3.execute(x);
    x = this->bn3.execute(x);
    x = this->relu.execute(x);
    x = this->downsample.execute(x);
    x = x + residual;
    x = this->relu.execute(x);
    return x;
}