#include "mesh_tensor.h"


MeshTensor::MeshTensor(
                torch::Tensor faces, 
                torch::Tensor feats,
                std::map<std::string, std::vector<float>> cache
                ){
    /*
    Parameters
    ------------
    faces: (N, F, 3) int32
        Array of triangular faces.
    feats: (N, C, F) float32
        Array of face features.
    Fs: (N,) int32, optional
        Array of number of faces in each mesh. 
        If not specified, Fs is set to n.
    cache: dict, optional
        things calculated from faces to avoid repeated calculation for the 
        same mesh.
    */
    
    this->faces = faces;
    this->feats = feats;
    this->Fs = Fs;
    this->cache = cache;
}

MeshTensor MeshTensor::updated(torch::Tensor new_feats){
    /*
    Return a new MeshTensor with its feats updated. 
        
    A shortcut to obtain a new MeshTensor with new features.
    */
    assert(new_feats.size(0) == this->feats.size(0));
    assert(new_feats.size(1) == this->feats.size(1));
    return MeshTensor(this->faces, new_feats, this->cache);
}

MeshTensor MeshTensor::inverseLoopPool(std::string op, std::optional<torch::Tensor> pooled_feats){
    /*
    Pooling with the inverse loop scheme.

    Parameters:
    ------------
    op: {'max', 'mean'}, optional
        Reduction method of pooling. The default is 'max'.
    pooled_feats: (N, C, F) float32, optional
        Specifying the feature after pooling.

    Returns:
    ------------
    MeshTensor after 4-to-1 face merge.
    */
    assert(op == "max" || op == "mean");
}
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
    return MeshTensor(feats, this->faces, this->cache);
}

MeshTensor MeshTensor::operator-(MeshTensor const& other) const{
    // subtracts the features of two meshes together
    torch::Tensor feats = this->feats - other.feats;
    return MeshTensor(feats, this->faces, this->cache);
}


// properties
std::vector<int> MeshTensor::shape(){
    auto shape = this->feats.sizes();
    return std::vector<int>(shape.begin(), shape.end());
}


int MeshTensor::V(){
    /* Maximum number of vertices in the mini-batch */

    auto it = this->cache.find("V");
    if (it != this->cache.end()){
        return this->cache["V"][0];
    } else{
        float V = this->faces.max().item<int>();
        this->cache["V"] = {V};
        return int(V);
    }
    
}


// std::vector<float> MeshTensor::Vs(){
//     /*
//     Number of vertices in each mesh. 
    
//     Returns
//     ------------
//     (N,) int32
//     */
//     auto it = this->cache.find("Vs");
//     if (it != this->cache.end()){
//         return this->cache["Vs"];
//     } else {
//         auto Vs = torch::amax(this->faces, 1).item<float>();
//         return std::vector<float>(Vs.begin<int>(), Vs.end<int>());
//     }

// }


// std::vector<int> MeshTensor::degrees(){
//     /*
//     Degrees of vertices.

//     Return:
//     ------------
//     (N, V) int32
//     */
//     auto it = this->cache.find("degrees");
//     if (it != this->cache.end()){
//         return this->cache["degrees"];
//     } else {
//         auto degrees = torch::zeros({this->faces.size(0), this->V()});
//         auto faces = this->faces;
//         auto Fs = this->Fs;
//         auto V = this->V();
//         for (int i=0; i<Fs; i++){
//             auto face = faces[i];
//             for (int j=0; j<3; j++){
//                 auto v = face[j].item<int>();
//                 degrees[i][v] += 1;
//             }
//         }
//         return std::vector<int>(degrees.begin<int>(), degrees.end<int>());
//     }
// }


std::vector<int> MeshTensor::FAF(){}
std::vector<int> MeshTensor::FAFP(){}
std::vector<int> MeshTensor::FAFN(){}