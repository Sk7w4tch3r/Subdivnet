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

    torch::Tensor y = meshTensor.feats;

    if (this->k_kernel_size == 1){
        torch::Tensor feats = meshTensor.feats;
        if (this->k_stride == 2){
            throw std::invalid_argument("not implemented");
            // int N = feats.size(0);
            // int C = feats.size(1);
            // int F = feats.size(2);
            // feats = torch::cat({feats, torch::zeros({N, C, 1})}, 2); // TODO: reindex 
            // meshTensor = meshTensor.inverseLoopPool("max", feats);
        }
        torch::Tensor y = this->conv1d->forward(feats);
    } else {
        std::unordered_map<givde::multiatlas::Index, std::vector<torch::Tensor>> neighorfeats;
        
        int K = 3; // triangular mesh
        torch::Tensor convFeats = torch::zeros({meshTensor.N, this->k_in_channels, meshTensor.F, K});


        // TODO this needs to be changed to support moving through hierarchy
        for (int i = 0; i < meshTensor.N; i++){ // for each mesh
            for (int j = 0; j < meshTensor.F; j++){ // for each face
                Index idx = meshTensor.faceIdxMap[j];
                IndexList neighbors = givde::multiatlas::neighbors(meshTensor.madggs, idx, givde::NeighborTypes::EDGE);
                for (int k = 0; k < K; k++){ // for each neighbor
                    Index neighbor = neighbors[k];
                    int neighborIdx = meshTensor.idxFaceMap[neighbor];
                    convFeats[i][0][j][k] = meshTensor.feats[i][0][neighborIdx];
                    convFeats[i][1][j][k] = meshTensor.feats[i][1][neighborIdx];
                    convFeats[i][2][j][k] = meshTensor.feats[i][2][neighborIdx];
                }                
            }
        }

        torch::Tensor y0 = meshTensor.feats; // N, C, F

        std::vector<torch::Tensor> features = {y0};
        features.push_back(convFeats.sum(3));
        auto idxArr = torch::cat({torch::tensor({static_cast<int64_t>(K - 1)}), torch::arange(K - 1)});
        features.push_back(torch::abs(convFeats.index({idxArr})-convFeats).sum(3));
        features.push_back(torch::abs(y0.unsqueeze(3) - convFeats).sum(3));
        torch::Tensor concat_feats = torch::cat(features, 1);

        auto convY = this->conv2d->forward(concat_feats);
        y = convY.view({meshTensor.N, this->k_out_channels, meshTensor.F});
    }
    return meshTensor.updated(y);
}

MeshTensor MeshPool::forward(MeshTensor meshTensor){
    return meshTensor.inverseLoopPool(this->k_op);
}

// MeshTensor MeshUnpool::forward(MeshTensor meshTensor, MeshTensor refMesh){
//     if (refMesh.faces.size(0) > 0){
//         meshTensor = meshTensor.loopUnPool(this->k_mode, refMesh.faces, refMesh.cache);
//     } else {
//         meshTensor = meshTensor.loopUnPool(this->k_mode);
//     }
//     return meshTensor;
// }


torch::Tensor MeshAdaptivePool::forward(MeshTensor meshTensor){
    auto y = meshTensor.feats; // the shape of feats is (N, C, F) and final shape should be (N, C)
    if (this->k_mode == "max"){
        y = std::get<0>(torch::max(y, 2));
    } else if (this->k_mode == "mean"){
        y = torch::mean(y, 2);
    } else {
        // raise error
        throw std::invalid_argument("mode must be either 'max' or 'mean'");
    }

    return y;
}


MeshTensor meshConcat(std::vector<MeshTensor> meshTensors){
    // concat feats of meshTensors into a single tensor
    std::vector<torch::Tensor> feats = meshTensors | ranges::views::transform([](MeshTensor meshTensor){return meshTensor.feats;}) | ranges::to<std::vector<torch::Tensor>>();
    torch::Tensor featsTensor = torch::cat(feats, 1);
    return meshTensors[0].updated(featsTensor);
}


MeshTensor meshAdd(MeshTensor meshTensor1, MeshTensor meshTensor2){
    // add feats of meshTensor1 and meshTensor2
    meshTensor1.feats = meshTensor1.feats + meshTensor2.feats;
    return meshTensor1;
}