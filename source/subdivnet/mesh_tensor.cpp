#include "mesh_tensor.h"


IndexList makeIndex(
    givde::multiatlas::DGGS const &madggs,
    givde::Resolution const& resolution) {

    IndexList workingSet = givde::multiatlas::getBaseCells(madggs);

    IndexList currentChildren;
    IndexList nextChildren;

    if (resolution == givde::Resolution(0)){
        // if resolution is 0, then we want to color the base cells
        return workingSet;
    } else {
        // Level 1
        for (auto const &cell : workingSet) {
            for (auto const &child : givde::multiatlas::children(madggs, cell)) {
                currentChildren.push_back(child);
            }
        }

        // Level 2+
        for (std::size_t i = 0; i < resolution - 1; ++i) {
            for (auto const &child : currentChildren) {
                for (auto const &grandChildren : givde::multiatlas::children(madggs, child)) {
                    nextChildren.push_back(grandChildren);
                }
            }
            nextChildren.swap(currentChildren);
            nextChildren.clear();
        }
    }
    return currentChildren;
}



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
    > {
    givde::i64 numFaces = idxs.size();

    givde::geometry::IGLGeometry mesh;
    auto vertSize = numFaces * 3;
    mesh.V        = givde::MatX<givde::f64>::Zero(vertSize, 3);
    mesh.N        = givde::MatX<givde::f64>::Zero(vertSize, 3);
    mesh.F        = givde::MatX<givde::i64>::Zero(numFaces, 3);
    mesh.C        = givde::MatX<givde::f64>::Zero(vertSize, 3);
    mesh.U        = givde::MatX<givde::f64>::Zero(vertSize, 3);
    mesh.D        = givde::MatX<givde::f64>::Zero(vertSize, 3);

    // for (Eigen::Index i = 0; i < mesh.C.rows(); ++i) {
    //     auto color = colors[i];
    //     mesh.C.row(i) = color.transpose();
    // }

    std::unordered_map<int, givde::multiatlas::Index> faceIdxMap;
    std::unordered_map<givde::multiatlas::Index, int> idxFaceMap;

    int idx = 0;
    for (auto const &child : idxs) {
        auto verts       = givde::multiatlas::cellToVertsCurved(dggs, child);
        givde::i64 v1Idx = (idx * 3) + 0;
        givde::i64 v2Idx = (idx * 3) + 1;
        givde::i64 v3Idx = (idx * 3) + 2;

        mesh.V.row(v1Idx) = verts.at(0).raw();
        mesh.V.row(v2Idx) = verts.at(1).raw();
        mesh.V.row(v3Idx) = verts.at(2).raw();

        mesh.D.row(v1Idx) = givde::Vec3<givde::f64>(1.0, 0.0, 0.0);
        mesh.D.row(v2Idx) = givde::Vec3<givde::f64>(0.0, 1.0, 0.0);
        mesh.D.row(v3Idx) = givde::Vec3<givde::f64>(0.0, 0.0, 1.0);

        mesh.F.row(idx) = givde::Vec3<givde::i64>(v1Idx, v2Idx, v3Idx);
        faceIdxMap[idx] = child;
        idxFaceMap[child] = idx;
        ++idx;
    }
    return std::make_tuple(mesh, std::make_tuple(faceIdxMap, idxFaceMap));
}


MeshTensor::MeshTensor(   
    torch::Tensor faces, 
    torch::Tensor feats,
    std::vector<int> Fs,
    std::unordered_map<std::string, torch::Tensor> cache){ 
    
    auto dggsIdxs = makeIndex(this->madggs, this->resolution);
    auto meshIdxs = idxsToMesh(this->madggs, dggsIdxs);
    this->dggsMesh = std::get<0>(meshIdxs);
    auto maps = std::get<1>(meshIdxs);
    this->faceIdxMap = std::get<0>(maps);
    this->idxFaceMap = std::get<1>(maps);
    
    this->faces = faces;
    this->feats = feats;
    this->Fs = Fs;
    this->cache = cache;
    this->N = this->faces.size(0);
    this->F = this->faces.size(1);
    this->C = this->feats.size(1);
}


MeshTensor MeshTensor::updated(torch::Tensor & new_feats){
    /*
    Return a new MeshTensor with its feats updated. 
        
    A shortcut to obtain a new MeshTensor with new features.
    */
    assert(new_feats.size(0) == this->feats.size(0));
    assert(new_feats.size(1) == this->feats.size(1));
    return MeshTensor(this->faces, new_feats, Fs, cache);
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
    std::vector<int> pooled_Fs = this->Fs | ranges::view::transform([](int F){return F/4;}) | ranges::to<std::vector<int>>();
    torch::Tensor pooled_faces = torch::zeros({this->faces.size(0), pooled_Fs[0], 3});
    
    // reindex

    if (pooled_feats.has_value()){
        assert(pooled_feats.value().size(0) == this->N);
        assert(pooled_feats.value().size(2) == this->F/4);
    } else {
        // reindex
        if (op == "max"){
            pooled_feats = torch::zeros({this->N, this->C, this->F/4});
        } else if (op == "mean"){
            pooled_feats = torch::zeros({this->N, this->C, this->F/4});
        } else {
            throw std::invalid_argument("op must be 'max' or 'mean'");
        }
    }
    return MeshTensor(pooled_faces, pooled_feats.value(), pooled_Fs, this->cache);
}


// torch::Tensor MeshTensor::loopSubdivision(){
//     torch::Tensor subdiv_faces = torch::zeros({this->N, this->F*4, 3});
//     for (int i = 0; i < this->N; i++){ // for each mesh
//         int V = this->faces[i].max().item<int>() + 1;
//         torch::Tensor face = this->faces[i];
//         int F = this->Fs[i];

//         torch::Tensor E = torch::cat(
//             {
//                 this->faces[]})
//     }
//     return subdiv_faces;
// }


// MeshTensor MeshTensor::loopUnPool(std::string mode, torch::Tensor ref_faces = torch::Tensor(), std::map<std::string, std::vector<float>> ref_cache = std::map<std::string, std::vector<float>>()){
//     std::vector<int> unpooled_Fs = this->Fs | ranges::view::transform([](int F){return F*4;}) | ranges::to<std::vector<int>>();
//     torch::Tensor unpooled_faces = torch::zeros({this->faces.size(0), unpooled_Fs[0], 3});
//     torch::Tensor unpooled_feats = torch::zeros({this->feats.size(0), this->feats.size(1), unpooled_Fs[0]});

//     if (ref_faces.size(0) > 0){
//         assert(ref_faces.size(0) == this->faces.size(0));
//         assert(ref_faces.size(1) == this->faces.size(1));
//         assert(ref_faces.size(2) == 3);
//         unpooled_faces = ref_faces;
//     } else {
//         unpooled_faces = this->loopSubdivision();
//     }


//     if (mode == "nearest"){
//         unpooled_feats = torch::cat({this->feats, this->feats, this->feats, this->feats}, 2);
//     } else if (mode == "bilinear"){

//     } else {
//         throw std::invalid_argument("mode must be 'nearest' or 'bilinear'");
//     }


//     return MeshTensor(ref_faces, this->feats, unpooled_Fs, ref_cache);
// }

// torch::Tensor MeshTensor::computeFaceAdjacencyFaces(){
//     torch::Tensor FAF = torch::zeros({this->faces.size(0), this->faces.sisubtractedFeatsze(1), 3});
//     for (int i = 0; i < this->N; i++){
//         int F = this->Fs[i];
//         torch::Tensor E = torch::cat({this->faces[i], torch::roll(this->faces[i], -1, 1)}, 1);

//         // E_hash = E.min(1) * E.max() + E.max(1)
//         torch::Tensor E_min = torch::stack({std::get<0>(E.min(1)), std::get<1>(E.min(1))}, 1);
//         torch::Tensor E_hash = E_min * E.max() + std::get<0>(E.max(1));
        
//         // S is index of sorted E_hash.
//         // Based on the construction rule of E,
//         //   1. S % F is the face id
//         //   2. S // F is the order of edge in F
//         torch::Tensor S = torch::argsort(E_hash);

//         // S[:, 0] is the face id of the first edge in E
//         // S[:, 1] is the face id of the second edge in E
//         S = S.reshape({F, 3, 2});

//         FAF[i, S[:, 0] % F, S[:, 0] / F] = S[:, 1] % F;
//         FAF[i, S[:, 1] % F, S[:, 1] / F] = S[:, 0] % F;
//     }
//     return FAF;
// }


torch::Tensor MeshTensor::computeFaceAdjacencyFaces(){
    // compute face adjacency faces
    // for each face in each mesh, find the adjacent faces (3 faces)

    torch::Tensor FAF = torch::zeros({this->faces.size(0), this->faces.size(1), 3});

    for (int i = 0; i < this->N; i++){ // for each mesh
        int F = this->Fs[i];
        for (int j = 0; j < F; j++){ // for each face
            if (this->faceIdxMap.find(j) != this->faceIdxMap.end()){
                // if the face is in the faceIdxMap
                auto idx = this->faceIdxMap[j];
                auto neighbors = givde::multiatlas::neighbors(this->madggs, idx, givde::NeighborTypes::EDGE);
                
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> MeshTensor::computeFaceAdjacencyReordered(){
    throw std::invalid_argument("not implemented");
    // TODO implement shift and roll
    torch::Tensor FAFN = this->FAF();
    torch::Tensor FAFP = this->FAF();
    return std::make_tuple(FAFN, FAFP);
}


torch::Tensor MeshTensor::dilatedFaceAdjacencies(int const& dilation){
    if (dilation <= 1){
        throw std::invalid_argument("dilation must be greater than 1");
    } else {
        // TODO implement the CUDA kernel
        throw std::invalid_argument("not implemented");
    }
    return torch::zeros({this->faces.size(0), this->faces.size(1), 3});
}


torch::Tensor MeshTensor::convolutionKernelPattern(int const& kernel_size, int const& dilation){
    if (kernel_size==3){
        if (dilation==1){
            return this->FAF();
        } else {
            return this->dilatedFaceAdjacencies(dilation);
        }
    } else if (kernel_size==5){
        throw std::invalid_argument("not implemented");
        if (dilation==1){
            return torch::cat(
                // FAF* is 3d tensor
                // 0: no meshes
                // 1: no of faces
                // 2: no of adjacent faces (3)
                {
                    this->FAFN().slice(2, 0, 1),
                    this->FAF().slice(2, 0, 1),
                    this->FAFP().slice(2, 0, 1),
                    this->FAFN().slice(2, 1, 2),
                    this->FAF().slice(2, 1, 2),
                    this->FAFP().slice(2, 1, 2),
                    this->FAFN().slice(2, 2, 3),
                    this->FAF().slice(2, 2, 3),
                    this->FAFP().slice(2, 2, 3)
                }, 2);
        } else {
            throw std::invalid_argument("not implemented");
        }
    } else {
        // TODO implement the CUDA kernel
    }
}

// operations
MeshTensor MeshTensor::operator+(MeshTensor & other) const{
    // adds the features of two meshes together
    torch::Tensor addedFeats = this->feats + other.feats;
    return MeshTensor(faces, addedFeats, Fs, cache);
}

MeshTensor MeshTensor::operator-(MeshTensor & other) const{
    // subtracts the features of two meshes together
    torch::Tensor subtractedFeats = this->feats - other.feats;
    return MeshTensor(faces, subtractedFeats, Fs, cache);
}


// properties


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


torch::Tensor MeshTensor::FAF(){
    if (this->cache.find("FAF") == this->cache.end()){
        this->cache["FAF"] = this->computeFaceAdjacencyFaces();
    } else {
        return this->cache["FAF"];
    }
}


torch::Tensor MeshTensor::FAFP(){
    if (this->cache.find("FAFP") == this->cache.end()){
        this->cache["FAFP"] = std::get<1>(this->computeFaceAdjacencyReordered());
    } else {
        return this->cache["FAFP"];
    }
}


torch::Tensor MeshTensor::FAFN(){
    if (this->cache.find("FAFN") == this->cache.end()){
        this->cache["FAFN"] = std::get<0>(this->computeFaceAdjacencyReordered());
    } else {
        return this->cache["FAFN"];
    }
}

