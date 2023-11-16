#include "mesh_tensor.h"


MeshTensor::MeshTensor(torch::Tensor faces, 
                torch::Tensor feats,
                int Fs,
                std::optional<std::map<int, int>> cache = std::nullopt){
    this->faces = faces;
    this->feats = feats;
    this->Fs = Fs;
    this->cache = cache;
}

MeshTensor MeshTensor::updated(torch::Tensor new_feats){
    assert(new_feats.size(0) == this->feats.size(0));
    assert(new_feats.size(1) == this->feats.size(1));
    return MeshTensor(this->faces, new_feats, this->Fs);
}

MeshTensor MeshTensor::inverseLoopPool(){}
MeshTensor MeshTensor::loopSubdivision(){}
MeshTensor MeshTensor::loopUnPool(){}
std::vector<int> computeFaceAdjacencyFaces(){}
std::vector<int> computeFaceAdjacencyReordered(){}
MeshTensor MeshTensor::dilatedFaceAdjacencies(){} // should return jit code result
MeshTensor MeshTensor::convolutionKernelPattern(){} // should return jit code result
std::vector<float> aggregateVertexFeatures(){}


// operations
MeshTensor MeshTensor::operator+(MeshTensor const& other) const{
    // adds the features of two meshes together
    torch::Tensor feats = this->feats + other.feats;
    return MeshTensor(feats, this->faces, this->Fs);
}

MeshTensor MeshTensor::operator-(MeshTensor const& other) const{
    // subtracts the features of two meshes together
    torch::Tensor feats = this->feats - other.feats;
    return MeshTensor(feats, this->faces, this->Fs);
}


// properties
std::vector<int> MeshTensor::shape(){
    auto shape = this->feats.sizes();
    return std::vector<int>(shape.begin(), shape.end());
}
int MeshTensor::V(){}
std::vector<int> MeshTensor::Vs(){}
std::vector<int> MeshTensor::degrees(){}
std::vector<int> MeshTensor::FAF(){}
std::vector<int> MeshTensor::FAFP(){}
std::vector<int> MeshTensor::FAFN(){}