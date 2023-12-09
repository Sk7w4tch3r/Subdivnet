#pragma once

#include <filesystem>

#include <torch/torch.h>
// include torch dataset class
#include <torch/data/datasets/base.h>
#include "mesh_tensor.h"
#include <igl/readOFF.h>
#include <igl/adjacency_list.h>
#include <Eigen/Geometry>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel>                          Polyhedron;
typedef Polyhedron::Vertex_handle                           Vertex_handle;
namespace PMP = CGAL::Polygon_mesh_processing;

struct MeshTorch {
    torch::Tensor V;
    torch::Tensor F;
    MeshTorch(torch::Tensor V, torch::Tensor F) : V(V), F(F) {}
};


torch::Tensor augment_points(torch::Tensor pts){
    // scale the points by a factor between 0.8 and 1.25
    pts = pts * torch::randn({1, 3}) * 0.25 + 1;
    // translation by a random vector with each component between -0.1 and 0.1
    pts = pts + (torch::randn({1, 3}) * 0.2 - 0.1);
    return pts;
}


MeshTorch randomize_mesh_orientation(MeshTorch mesh){
    // randomly rotate the orientation of mesh by 0, 90, 180, or 270 degrees
    // in randomly picked order of x, y, and z axes using eigen library
    std::string axis_seq = "xyz";
    std::random_shuffle(axis_seq.begin(), axis_seq.end());

    // Generate random angles
    std::vector<double> angles;
    for (int i = 0; i < 3; ++i) {
        angles.push_back(torch::randint(4, {1}).item<int>() * 90.0);
    }

    // Convert degrees to radians
    for (auto& angle : angles) {
        angle *= M_PI / 180.0;
    }

    // Construct rotation matrix
    Eigen::AngleAxis<double> rotation_x(angles[0], Eigen::Vector3d::Unit(axis_seq[0] - 'x'));
    Eigen::AngleAxis<double> rotation_y(angles[1], Eigen::Vector3d::Unit(axis_seq[1] - 'y'));
    Eigen::AngleAxis<double> rotation_z(angles[2], Eigen::Vector3d::Unit(axis_seq[2] - 'z'));

    Eigen::Quaternion<double> quaternion = rotation_z * rotation_y * rotation_x;
    Eigen::Matrix3d rotation_matrix = quaternion.matrix();

    // Convert Eigen matrix to torch::Tensor
    torch::Tensor rotation_matrix_tensor = torch::from_blob(rotation_matrix.data(), {3, 3}, torch::kDouble);

    // Apply rotation to mesh vertices
    mesh.V = torch::matmul(mesh.V, rotation_matrix_tensor);

    return mesh;
};

MeshTorch random_scale(MeshTorch mesh){
    // randomize the scale of the mesh vertex positions by a factor between 0.9 and 1.1
    mesh.V = mesh.V * torch::randn({1, 3}) * 0.1 + 1;
    return mesh;
};


MeshTorch mesh_normalize(MeshTorch mesh){
    // normalize vertex positions
    mesh.V = mesh.V - torch::mean(mesh.V, 0);
    return mesh;
};

std::tuple<torch::Tensor, std::vector<torch::Tensor>> load_mesh(
    std::string path,
    std::optional<bool> normalize,
    std::optional<std::vector<std::string>> augments,
    std::optional<std::vector<std::string>> request){

    // load a mesh from a file
    Eigen::MatrixXd VEigen;
    Eigen::MatrixXi FEigen;

    igl::readOFF(path, VEigen, FEigen);
    
    // convert eigen matrices to torch tensors
    torch::Tensor F = torch::from_blob(FEigen.data(), {FEigen.rows(), FEigen.cols()}, torch::kInt32);
    torch::Tensor V = torch::from_blob(VEigen.data(), {VEigen.rows(), VEigen.cols()}, torch::kFloat32);

    // MeshTensor mesh(F, V, 0, std::map<std::string, std::vector<float>>());
    MeshTorch mesh(V, F);

    if (augments.has_value()){
        for (auto augment : augments.value()){
            if (augment == "orient"){
                mesh = randomize_mesh_orientation(mesh);
            } else if (augment == "scale"){
                mesh = random_scale(mesh);
            } else {
                throw std::invalid_argument("Invalid augment");
            }
        }
    }

    if (normalize.has_value()){
        if (normalize.value()){
            mesh = mesh_normalize(MeshTorch(V, F));
        }
    }
    
    // get the number of faces
    int Fs = F.size(0);

    std::vector<torch::Tensor> feats;

    if (request.has_value()){
        for (auto req : request.value()){
            if (req == "area"){
                feats.push_back(torch::Tensor());
            } else if (req == "normal"){
                torch::Tensor vertex_normals = torch::zeros({VEigen.rows(), 3}, torch::kFloat32);
                // CGAL::Polygon_mesh_processing::compute_vertex_normals(mesh, vertex_normals);

                torch::Tensor face_normals = torch::zeros({Fs, 3}, torch::kFloat32);
                // CGAL::Polygon_mesh_processing::compute_face_normals(mesh, face_normals);
                
                feats.push_back(torch::Tensor());
            } else if (req == "center"){
                torch::Tensor face_center = torch::zeros({Fs, 3}, torch::kFloat32);
                feats.push_back(face_center);
            } else if (req == "face_angles"){
                feats.push_back(torch::Tensor());
            } else if (req == "curves"){
                torch::Tensor face_curvs = torch::zeros({Fs, 3}, torch::kFloat32);
                feats.push_back(face_curvs);
            } else {
                throw std::invalid_argument("Invalid request");
            }
        }
    }
    
    torch::Tensor feats_tensor = torch::stack(feats, 1);

    // MeshTensor mesh = MeshTensor(F, feats_tensor, Fs, std::map<std::string, std::vector<float>>());

    return {F, feats};
}


class ClassificationDataset : public torch::data::datasets::Dataset<ClassificationDataset> {
    using torch::data::datasets::Dataset<ClassificationDataset>::Dataset;
    public:
        std::string path;
        int k_batch_size;
        bool k_train;
        bool k_shuffle;
        bool k_augment;
        bool k_in_memory;

        std::vector<std::string> request;
        std::vector<std::string> augments;
        std::string mode;
        std::vector<std::string> feats;

        std::vector<std::string> mesh_paths;
        std::vector<std::string> labels;
        std::map<std::string, int> label_to_idx;

        int total_len = 0;

        ClassificationDataset(
            std::string const& dataroot, 
            int batch_size,
            bool train,
            bool shuffle,
            bool augment,
            bool in_memory){
            
            this->path = dataroot;
            this->k_batch_size = batch_size;
            this->k_train = train;
            this->k_shuffle = shuffle;
            this->k_augment = augment;
            this->k_in_memory = in_memory;

            this->browse_dataroot();
            this->total_len = this->mesh_paths.size();
        }

        void browse_dataroot(){
            // browse the dataroot and get the list of all the meshes
            // iterate over subdirectories of dataroot, use the subdirectory name as the label
            // and add mesh paths in each subdirectory to the mesh_paths list
            for (const auto & entry : std::filesystem::directory_iterator(this->path)){
                // if label is a directory
                if (std::filesystem::is_directory(entry.path())){
                    std::string label = entry.path().filename().string();
                    for (const auto & mesh_path : std::filesystem::directory_iterator(entry.path())){
                        // if mesh_path is a file
                        if (std::filesystem::is_regular_file(mesh_path.path()))
                        {
                            this->mesh_paths.push_back(mesh_path.path().string());
                            this->labels.push_back(label);
                            // convert label to index
                            if (this->label_to_idx.find(label) == this->label_to_idx.end()){
                                this->label_to_idx[label] = this->label_to_idx.size();
                            }
                            this->targets = torch::cat({this->targets, torch::tensor(this->label_to_idx[label])});
                        }
                    }
                }
            }

        }


        torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override {
            std::tuple<torch::Tensor, std::vector<torch::Tensor>> loaded_mesh = load_mesh(this->mesh_paths[index], true, this->augments, this->feats);
            torch::Tensor data = torch::cat({std::get<0>(loaded_mesh), torch::stack(std::get<1>(loaded_mesh), 1)}, 1);
            torch::Tensor label = this->targets[index];
            return { data, label };
        }

        torch::optional<size_t> size() const override {
            return this->data.size();
        }


    private:
        std::vector<MeshTensor> data;
        torch::Tensor targets;
};

