#pragma once
#include <givde/lib/dggs/multiatlas.h>


#include "utils.h"
#include "dataset.h"


void convertMNISTToDGGSCereal(
    std::string const& mnistDataPath,
    std::string const& mnistLabelPath,
    givde::multiatlas::DGGS const& dggs,
    givde::Resolution resolution){
    
    // load mnist dataset
    auto images = read_mnist(mnistDataPath);
    auto labels = read_mnist_labels(mnistLabelPath);


    std::cout << "size of images: " << images.size() << std::endl;

    IndexList dggsIdxs = makeIndex(dggs, resolution);
    auto meshIdxs = idxsToMesh(dggs, dggsIdxs);
    auto mesh = std::get<0>(meshIdxs);
    auto maps = std::get<1>(meshIdxs);
    auto faceIdxMap = std::get<0>(maps);
    auto idxFaceMap = std::get<1>(maps);

    for (std::size_t i = 0; i < images.size(); ++i) {
        auto image = images[i];
        auto label = labels[i];
        
        getIdxsColors(mesh, image, label);
        {
            std::string filename = "data/MNISTDGGS/" + std::to_string(i) + ".cereal";
            std::ofstream ofs(filename, std::ios::binary);
            cereal::BinaryOutputArchive archive(ofs);
            archive(mesh);
        }
    }
}