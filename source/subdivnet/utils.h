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

MeshTensor toMeshTensor(torch::Tensor faces, torch::Tensor features) {
    MeshTensor mesh_tensor = MeshTensor(faces, features, std::map<std::string, std::vector<float>>());
    return mesh_tensor;
}


class ClassificationMajorityVoting{

};

