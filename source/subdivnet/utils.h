#include <vector>
#include <torch/torch.h>

#include "mesh_tensor.h"

std::vector<std::vector<int>> segment_colors = {
    {0, 114, 189},
    {217, 83, 26},
    {238, 177, 32},
    {126, 47, 142},
    {117, 142, 48},
    {76, 190, 238},
    {162, 19, 48},
    {240, 166, 202}
};

MeshTensor toMeshTensor(torch::Tensor vertices, torch::Tensor faces, torch::Tensor features) {
    MeshTensor mesh_tensor;
    mesh_tensor.vertices = vertices;
    mesh_tensor.faces = faces;
    mesh_tensor.features = features;
    return mesh_tensor;
}


void saveResults(mesh_infos, preds, labels, name){
    continue;
}

void updateLabelAccuracy(preds, labels, acc){
    continue;
}

std::vector<float> computeOriginalAccuracy(mesh_infos, preds, labels){
    continue;
}

class ClassificationMajorityVoting{

};

class SegmentationMajorityVoting{

};


