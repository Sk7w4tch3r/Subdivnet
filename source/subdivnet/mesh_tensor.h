#pragma once

#include <map>
#include <optional>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>

#include <givde/lib/dggs/multiatlas.h>

#include <torch/torch.h>

// #include "utils.h"

#include "parallel.cu"


using Index = givde::multiatlas::Index;
using IndexList = std::vector<givde::multiatlas::Index>;


class MeshTensor {
    /*
    A MeshTensor object stores a batch of triangular meshes with 
    multi-dimensional arrays.

    All faces are stored in a 3-dimensional tensor. To support a batch of 
    variable number of faces, an addtional array Fs is used to hold every mesh's
    number of faces. 
    */
    public:
        /*
        Parameters
        ------------
        faces: (N, F, 3) int32
            Array of triangular faces.
        feats: (N, C, F) float32
            Array of face features.
        Fs: (N, 1) int32, optional
            Array of number of faces in each mesh. 
            If not specified, Fs is set to n.
        cache: dict, optional
            things calculated from faces to avoid repeated calculation for the 
            same mesh.
        */
        // DGGSTensor dggsTensor;
        givde::multiatlas::DGGS madggs = givde::multiatlas::DGGS(
            givde::geometry::icosahedron(),
            givde::multiatlas::LatLngLookupGridArgs{10, 10},
            givde::multiatlas::NormalProjection{}
        );
        std::unordered_map<int, givde::multiatlas::Index> faceIdxMap;
        std::unordered_map<givde::multiatlas::Index, int> idxFaceMap;
        givde::Resolution resolution = givde::Resolution(5);
        givde::geometry::IGLGeometry dggsMesh;

        torch::Tensor faces;
        torch::Tensor feats;
        std::vector<int> Fs;
        std::unordered_map<std::string, torch::Tensor> cache;
        int N, F, C;

        MeshTensor(
            torch::Tensor faces, 
            torch::Tensor feats,
            std::vector<int> Fs,
            std::unordered_map<std::string, torch::Tensor> cache);


        MeshTensor updated(torch::Tensor & tensor);
        MeshTensor inverseLoopPool(
            std::string op="max", 
            std::optional<torch::Tensor> pooled_feats=std::nullopt
        );
        // torch::Tensor loopSubdivision();
        // MeshTensor loopUnPool(
        //     std::string mode, 
        //     torch::Tensor ref_faces = torch::Tensor(), 
        //     std::map<std::string, std::vector<float>> ref_cache 
        //                     = std::map<std::string, std::vector<float>>());
        torch::Tensor computeFaceAdjacencyFaces();
        std::tuple<torch::Tensor, torch::Tensor> computeFaceAdjacencyReordered();
        torch::Tensor dilatedFaceAdjacencies(int const& dilation); // should return jit code result
        torch::Tensor convolutionKernelPattern(int const& kernel_size=3, int const& dilation=1); // should return jit code result
        std::vector<float> aggregateVertexFeatures();

        // operations
        MeshTensor operator+(MeshTensor & other) const;
        MeshTensor operator-(MeshTensor & other) const;
        

        // properties
        std::vector<int> shape(){
            auto shape = this->faces.sizes();
            return std::vector<int>(shape.begin(), shape.end());
        };


        // int V(){
        //     if (this->cache.find("V") == this->cache.end()){
        //         this->cache["V"] = {(float)this->faces.max().item<int>()};
        //     }
        //     return this->cache["V"].item<int>();
        // };

        // std::vector<int> Vs(){
        //     if (this->cache.find("Vs") == this->cache.end()){
        //         auto Vs = torch::amax(this->faces, 1).item<float>();
        //         return std::vector<float>(Vs.begin<int>(), Vs.end<int>());
        //         this->cache["Vs"] = {(float)this->faces.max(1).values.max(0).values.item<int>()};
        //     }
        //     return this->cache["Vs"];
        // };


        torch::Tensor FAF();
        torch::Tensor FAFP();
        torch::Tensor FAFN();
};



IndexList makeIndex(
    givde::multiatlas::DGGS const &madggs,
    givde::Resolution const& resolution);


auto idxsToMesh(
    givde::multiatlas::DGGS const &dggs,
    IndexList const &idxs
    // std::vector<givde::Vec3<givde::f64>> const &colors
) -> std::tuple<
    givde::geometry::IGLGeometry, 
    std::tuple<
        std::unordered_map<int, givde::multiatlas::Index>, 
        std::unordered_map<givde::multiatlas::Index, int>
        >
    >;