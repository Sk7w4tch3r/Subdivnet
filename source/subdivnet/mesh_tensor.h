#pragma once

#include <map>
#include <optional>

#include <torch/torch.h>


class MeshTensor {
    /*
    A MeshTensor object stores a batch of triangular meshes with 
    multi-dimensional arrays.

    All faces are stored in a 3-dimensional tensor. To support a batch of 
    variable number of faces, an addtional array Fs is used to hold every mesh's
    number of faces. 
    */
    public:
        torch::Tensor faces;
        torch::Tensor feats;
        int Fs;
        std::map<std::string, std::vector<float>> cache;

        MeshTensor(torch::Tensor faces, 
                    torch::Tensor feats,
                    std::map<std::string, std::vector<float>> cache);


        MeshTensor updated(torch::Tensor tensor);
        MeshTensor inverseLoopPool(
            std::string op="max", 
            std::optional<torch::Tensor> pooled_feats=std::nullopt
        );
        MeshTensor loopSubdivision();
        MeshTensor loopUnPool();
        std::vector<int> computeFaceAdjacencyFaces();
        std::vector<int> computeFaceAdjacencyReordered();
        MeshTensor dilatedFaceAdjacencies(); // should return jit code result
        MeshTensor convolutionKernelPattern(); // should return jit code result
        std::vector<float> aggregateVertexFeatures();

        // operations
        MeshTensor operator+(MeshTensor const& other) const;
        MeshTensor operator-(MeshTensor const& other) const;
        

        // properties
        std::vector<int> shape();
        int V();
        std::vector<float> Vs();
        std::vector<int> degrees();
        std::vector<int> FAF();
        std::vector<int> FAFP();
        std::vector<int> FAFN();
};