#pragma once
#include <map>

#include <optional>
#include <torch/torch.h>


class MeshTensor {
    public:
        torch::Tensor faces;
        torch::Tensor feats;
        int Fs;
        std::optional<std::map<int, int>> cache;

        MeshTensor(torch::Tensor faces, 
                    torch::Tensor feats,
                    int Fs,
                    std::optional<std::map<int, int>> cache);


        MeshTensor updated(torch::Tensor tensor);
        MeshTensor inverseLoopPool();
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
        std::vector<int> Vs();
        std::vector<int> degrees();
        std::vector<int> FAF();
        std::vector<int> FAFP();
        std::vector<int> FAFN();
};