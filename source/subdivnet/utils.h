#pragma once

#include <math.h>
#include <algorithm>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

#include <torch/torch.h>

#include "mesh_tensor.h"

#include <cereal/archives/binary.hpp>
#include <givde/geometry/cereal/types.h>


using IndexList = std::vector<givde::multiatlas::Index>;
using Index = givde::multiatlas::Index;

MeshTensor toMeshTensor(torch::Tensor faces, torch::Tensor features) {
    MeshTensor mesh_tensor = MeshTensor(faces, features, {}, {});
    return mesh_tensor;
}

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<Eigen::MatrixXd> read_mnist(std::string const& path){
    std::vector<Eigen::MatrixXd> mnist_images;
    std::ifstream file (path);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        for(int i=0;i<number_of_images;++i){
            mnist_images.push_back(Eigen::MatrixXd(n_rows,n_cols));
            for(int r=0;r<n_rows;++r){
                for(int c=0;c<n_cols;++c){
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    mnist_images[i](r,c)= (double)temp;
                }
            }
        }
    }
    return mnist_images;
}


std::vector<int> read_mnist_labels(std::string const& filePath){
    std::vector<int> mnist_labels;
    std::ifstream file (filePath);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        if (magic_number!=2049) throw std::runtime_error("Invalid MNIST label file!");
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        for(int i=0;i<number_of_images;++i){
            char temp;
            file.read((char*)&temp,sizeof(temp));
            mnist_labels.push_back((int)temp);
        }
    }
    return mnist_labels;
}



// save mesh to file
void saveDggsToFile(
    givde::geometry::IGLGeometry const& mesh,
    std::string const& path) {
    std::ofstream ofs(path, std::ios::binary);
    cereal::BinaryOutputArchive archive(ofs);
    archive(mesh);
}

givde::geometry::IGLGeometry readDggsFromFile(std::string const& path) {
    std::ifstream ifs(path, std::ios::binary);
    cereal::BinaryInputArchive archive(ifs);
    givde::geometry::IGLGeometry mesh;
    archive(mesh);
    return mesh;
}