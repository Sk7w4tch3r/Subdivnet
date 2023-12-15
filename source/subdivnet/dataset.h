#pragma once

#include <filesystem>

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <igl/readOFF.h>
#include <igl/adjacency_list.h>
#include <Eigen/Geometry>

#include <cereal/archives/binary.hpp>
#include <givde/geometry/cereal/types.h>

#include "mesh_tensor.h"
#include "utils.h"


struct MeshTorch {
    torch::Tensor V;
    torch::Tensor F;
    MeshTorch(torch::Tensor V, torch::Tensor F) : V(V), F(F) {}
};


void mapMnistToMesh(
    givde::geometry::IGLGeometry &mesh,
    const Eigen::MatrixXd& mnistImage,
    double sphereRadius) {

    auto vertices = mesh.V;
    auto faces = mesh.F;
    
    int numFaces = faces.rows();
    Eigen::VectorXd faceValues(numFaces);

    // #pragma omp parallel for
    for (int i = 0; i < numFaces; ++i) {
        // Calculate the centroid of the face
        Eigen::Vector3d centroid = (vertices.row(faces(i, 0)) + vertices.row(faces(i, 1)) + vertices.row(faces(i, 2))) / 3.0;

        // Convert centroid to spherical coordinates
        double theta = std::atan2(centroid(1), centroid(0));
        double phi = std::acos(centroid(2) / sphereRadius);

        // Map spherical coordinates to image coordinates
        int col = static_cast<int>((theta + M_PI) / (2 * M_PI) * mnistImage.rows());
        int row = static_cast<int>((phi + 0.5 * M_PI) / M_PI * mnistImage.cols());

        // find the closest pixel
        int minDist = std::numeric_limits<int>::max();
        int minRow = 0;
        int minCol = 0;
        for (int r = row - 1; r <= row + 1; ++r) {
            for (int c = col - 1; c <= col + 1; ++c) {
                if (r >= 0 && r < mnistImage.rows() && c >= 0 && c < mnistImage.cols()) {
                    int dist = std::abs(r - row) + std::abs(c - col);
                    if (dist < minDist) {
                        minDist = dist;
                        minRow = r;
                        minCol = c;
                    }
                }
            }
        }

        // Convert the image coordinates to a pixel index
        int pixelIndex = minRow * mnistImage.cols() + minCol;

        // Set the face value to the pixel value
        faceValues(i) = mnistImage(minRow, minCol);
    }

    // Normalize the face values
    faceValues = faceValues / faceValues.maxCoeff();

    // update the mesh colors
    for (int i = 0; i < numFaces; ++i) {
        Eigen::Vector3d color(faceValues(i), faceValues(i), faceValues(i));
        mesh.C.row(i) = color.transpose();
    }
}

// change C color attribute of mesh inplace
void getIdxsColors(
    givde::geometry::IGLGeometry &mesh,
    Eigen::MatrixXd const& image,
    int const& label) {
    
    mapMnistToMesh(mesh, image, 1.0);
    // save label to mesh.D
    mesh.D = Eigen::MatrixXd::Zero(1, 1);
    mesh.D(0, 0) = label;
}


struct DGGSTensor {
    givde::geometry::IGLGeometry & mesh;
    std::unordered_map<int, givde::multiatlas::Index> const& faceIdxMap;
    std::unordered_map<givde::multiatlas::Index, int> const& idxFaceMap;
    DGGSTensor(
        givde::geometry::IGLGeometry mesh, 
        std::unordered_map<int, givde::multiatlas::Index> faceIdxMap,
        std::unordered_map<givde::multiatlas::Index, int> idxFaceMap) : 
            mesh(mesh), 
            faceIdxMap(faceIdxMap),
            idxFaceMap(idxFaceMap) {}
};


DGGSTensor load_dggs_mesh(
    std::string const& path,
    givde::multiatlas::DGGS const& dggs,
    givde::Resolution const& resolution){
    
    IndexList dggsIdxs = makeIndex(dggs, resolution);
    auto meshIdxs = idxsToMesh(dggs, dggsIdxs);
    auto mesh = std::get<0>(meshIdxs);
    auto maps = std::get<1>(meshIdxs);
    auto faceIdxMap = std::get<0>(maps);
    auto idxFaceMap = std::get<1>(maps);

   
    mesh = readDggsFromFile(path);

    // clip mesh.C from shape (no_vertices, 3) to (no_faces, 3)
    mesh.C = mesh.C.topRows(mesh.F.rows());

    // getIdxsColors(mesh, image, label);
    

    // save and load dggs mesh

    // std::ofstream ofs("data/dggs_mesh.cereal", std::ios::binary);
    // cereal::BinaryOutputArchive archive(ofs);
    // archive(mesh);
    

    // torch::Tensor V = torch::from_blob(mesh.V.data(), {mesh.V.rows(), mesh.V.cols()}, torch::kDouble).clone();
    // torch::Tensor F = torch::from_blob(mesh.F.data(), {mesh.F.rows(), mesh.F.cols()}, torch::kInt32).clone();
    // torch::Tensor C = torch::from_blob(mesh.C.data(), {mesh.C.rows(), mesh.C.cols()}, torch::kDouble).clone();
    // torch::Tensor D = torch::from_blob(mesh.D.data(), {mesh.D.rows(), mesh.D.cols()}, torch::kDouble).clone();

    // std::cout << "loading mnist images" << std::endl;
    // std::vector<Eigen::MatrixXd mnistDataset = loadMNISTImages("data/t10k-images-idx3-ubyte");
    // std::cout << "mnistImage size: " << mnistImage[0].rows() << std::endl;

    // double sphereRadius = 1;
    // auto meshMNISTDataset = mnistDataset | ranges::view::transform([&](auto const& mnistImage){
    //     auto meshMNIST = mapMnistToMesh(mesh.V, mesh.F, mnistImage, sphereRadius);
    //     return meshMNIST;
    // }) | ranges::to<std::vector<givde::geometry::IGLGeometry>>();
    
    // mesh.D = faceValues;
    // std::ofstream ofs("data/dggs_mesh.cereal", std::ios::binary);
    // cereal::BinaryOutputArchive archive(ofs);
    // archive(mesh);

    // std::cout << "mesh.D " << mesh.D << std::endl;
    // // print maximum value of mesh.C
    // std::cout << "mesh.C max: " << mesh.C.maxCoeff() << std::endl;

    // // print maximum value of tensor C
    // std::cout << "C max: " << C.max().item<float>() << std::endl;
    // // print value of D
    // std::cout << "D: " << D << std::endl;
    // std::cout << "Face 0: " << F[0] << std::endl;
    // std::cout << "Face 0 from mesh tensor: " << mesh.F.row(0) << std::endl;
    return DGGSTensor(mesh, faceIdxMap, idxFaceMap);
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
    torch::Tensor rotation_matrix_tensor = torch::from_blob(rotation_matrix.data(), {3, 3}, torch::kFloat32);

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

std::tuple<torch::Tensor, torch::Tensor> load_mesh(
    std::string path,
    std::optional<bool> normalize,
    std::optional<std::vector<std::string>> augments,
    std::optional<std::vector<std::string>> request){

    // load a mesh from a file
    Eigen::MatrixXd VEigen;
    Eigen::MatrixXi FEigen;

    igl::readOFF(path, VEigen, FEigen);
    // transpose faces
    FEigen.transposeInPlace();
    VEigen.transposeInPlace();
    // convert eigen matrices to torch tensors
    torch::Tensor F = torch::from_blob(FEigen.data(), {FEigen.cols(), FEigen.rows()}, torch::kInt32).clone();
    torch::Tensor V = torch::from_blob(VEigen.data(), {VEigen.cols(), VEigen.rows()}, torch::kDouble).clone();    

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

    return {F, feats_tensor};
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
            std::tuple<torch::Tensor, torch::Tensor> loaded_mesh = load_mesh(this->mesh_paths[index], true, this->augments, this->feats);
            torch::Tensor data = torch::cat({std::get<0>(loaded_mesh), torch::stack(std::get<1>(loaded_mesh), 1)}, 1);
            torch::Tensor label = this->targets[index];
            return { data, label };
        }

        torch::optional<size_t> size() const override {
            return this->targets.size(0);
        }


    private:
        // std::vector<MeshTensor> data;
        torch::Tensor targets;
};


template <typename Faces = torch::Tensor, typename Feats = torch::Tensor, typename Labels = torch::Tensor>
struct Example{
    using FaceType = Faces;
    using FeatsType = Feats;
    using LabelsType = Labels;

    Faces faces;
    Feats feats;
    Labels labels;

    Example() = default;
    Example(Faces faces, Feats feats, Labels labels) : faces(std::move(faces)), feats(std::move(feats)), labels(std::move(labels)) {}
};

template <typename ExampleType>
struct Stack : public torch::data::transforms::Collation<ExampleType> {
    ExampleType apply_batch(std::vector<ExampleType> examples) override {
        std::vector<torch::Tensor> faces, feats, labels;

        faces.reserve(examples.size());
        feats.reserve(examples.size());
        labels.reserve(examples.size());

        for (auto& example : examples) {
            faces.push_back(std::move(example.faces));
            feats.push_back(std::move(example.feats));
            labels.push_back(std::move(example.labels));
        }

        return ExampleType(
            torch::stack(faces, 0),
            torch::stack(feats, 0),
            torch::stack(labels, 0)
        );
    }
};


using DGGSExample = Example<torch::Tensor, torch::Tensor, torch::Tensor>;

class ClassificationDGGSDataset : public torch::data::datasets::Dataset<ClassificationDGGSDataset, DGGSExample> {
    using torch::data::datasets::Dataset<ClassificationDGGSDataset, DGGSExample>::Dataset;
    public:
        std::string path;
        int k_batch_size;
        bool k_train;
        bool k_shuffle;
        bool k_in_memory;

        std::string mode;
        std::vector<std::string> feats;

        std::vector<std::string> mesh_paths;
        std::vector<std::string> labels;
        std::map<std::string, int> label_to_idx;

        givde::multiatlas::DGGS k_dggs = givde::multiatlas::DGGS(
            givde::geometry::icosahedron(),
            givde::multiatlas::LatLngLookupGridArgs{10, 10},
            givde::multiatlas::NormalProjection{}
        );
        givde::Resolution k_resolution;

        int total_len = 0;

        ClassificationDGGSDataset(
            std::string dataroot, 
            int batch_size,
            bool train,
            bool shuffle,
            bool in_memory,
            givde::Resolution resolution){
            
            this->path = dataroot;
            this->k_batch_size = batch_size;
            this->k_train = train;
            this->k_shuffle = shuffle;
            this->k_in_memory = in_memory;

            // this->k_dggs = dggs;
            this->k_resolution = resolution;

            this->browse_dataroot();
            this->total_len = this->mesh_paths.size();
        }

        // virtual ~ClassificationDGGSDataset();

        void browse_dataroot(){
            for (const auto & entry : std::filesystem::directory_iterator(this->path)){
                if (std::filesystem::is_regular_file(entry.path())){
                    this->mesh_paths.push_back(entry.path().string());
                }
            }
        }


        DGGSExample get(size_t index) override {
            auto dggsTensor = load_dggs_mesh(this->mesh_paths[index], this->k_dggs, this->k_resolution);
            auto label = torch::tensor(dggsTensor.mesh.D(0, 0));

            torch::Tensor data = torch::from_blob(dggsTensor.mesh.C.data(), {dggsTensor.mesh.C.rows(), dggsTensor.mesh.C.cols()}, torch::kDouble).clone();


            return {data, data, label};
        }


        // implement batch loading
        // torch::data::Example<torch::Tensor, torch::Tensor> get_batch(size_t index) override {
        // }


        torch::optional<size_t> size() const override {
            return this->mesh_paths.size();
        }


    private:
        // std::vector<MeshTensor> data;

};

