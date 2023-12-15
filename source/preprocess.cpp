// clang-format off

#include <span>
#include <variant>

#include <torch/torch.h>

#include <igl/readOFF.h>
#include <igl/adjacency_list.h>
#include <igl/edges.h>

#include "subdivnet/network.h"
#include "subdivnet/dataset.h"
#include "subdivnet/mesh_tensor.h"
#include "subdivnet/convert.h"


auto main() -> int {
    
    givde::multiatlas::DGGS madggs = givde::multiatlas::DGGS(
        givde::geometry::icosahedron(),
        givde::multiatlas::LatLngLookupGridArgs{10, 10},
        givde::multiatlas::NormalProjection{}
    );
    std::cout << "DGGS created" << std::endl;
    
    convertMNISTToDGGSCereal(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
        madggs,
        givde::Resolution(5)
    );
    std::cout << "MNIST converted to DGGS" << std::endl;

    // auto mesh = readDggsFromFile("data/MNISTDGGS/9998.cereal");
    // std::cout << "mesh read from file" << std::endl;
    // std::cout << mesh.F.size() << std::endl;
    // std::cout << mesh.D(0, 0) << std::endl;

    // auto feats = mesh.C.block(0, 0, mesh.F.rows(), 3);


    return 0;
}
