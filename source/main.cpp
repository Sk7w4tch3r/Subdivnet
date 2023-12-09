// clang-format off

#include <span>
#include <variant>

#include <torch/torch.h>

// #include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

// #include <CGAL/Polyhedron_3.h>
// #include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

#include <igl/readOFF.h>
#include <igl/adjacency_list.h>
#include <igl/edges.h>
// #include <igl/opengl/glfw/Viewer.h>

#include "subdivnet/network.h"

// typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
// typedef CGAL::Polyhedron_3<Kernel>                          Polyhedron;
// typedef Polyhedron::Vertex_handle                           Vertex_handle;
// namespace PMP = CGAL::Polygon_mesh_processing;


// clang-format on


auto main() -> int {
    

    // const std::string filename = CGAL::data_file_path("/home/hesam/cpsc601/project/bg-cnn-cli/data/apple_0-0.obj");
    
    // Polyhedron mesh;
    
    // if(!PMP::IO::read_polygon_mesh(filename, mesh) || !CGAL::is_triangle_mesh(mesh))
    // {
    //     std::cerr << "Invalid input." << std::endl;
    //     return 1;
    // }


    // // print the number of vertices and faces

    // std::cout << "The mesh has " << num_vertices(mesh) << " vertices and "
    //           << num_faces(mesh) << " faces" << std::endl;

    // save faces in a tensor
    // torch::Tensor faces = torch::zeros({static_cast<int64_t>(num_faces(mesh)), 3}, torch::kInt32);
    // int i = 0;
    // iterate over the mesh faces
    // for (Polyhedron::Facet_const_iterator fit = mesh.facets_begin(); fit != mesh.facets_end(); ++fit)
    // {
    //     Polyhedron::Halfedge_around_facet_const_circulator circ = fit->facet_begin();
    //     faces[i][0] = circ->vertex()->index();
    //     faces[i][1] = circ->next()->vertex()->index();
    //     faces[i][2] = circ->next()->next()->vertex()->index();
    //     i++;
    // }

    MeshNet net;

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    // Load a mesh in OFF format
    igl::readOFF("data/apple_0-0.off", V, F);
    
    
// Get the edge list
    Eigen::MatrixXi E;
    igl::edges(F, E);

    // Build an adjacency list for faces based on shared edges
    std::vector<std::vector<int>> face_adjacency_list(F.rows());
    for (int i = 0; i < E.rows(); ++i)
    {
        for (int j = 0; j < E.cols(); ++j)
        {
            int face1 = E(i, j);
            int face2 = E(i, (j + 1) % E.cols());

            if (face1 != face2)
            {
                face_adjacency_list[face1].push_back(face2);
                face_adjacency_list[face2].push_back(face1);
            }
        }
    }

    // Print neighboring faces for each face based on shared edges
    for (int i = 0; i < 500; ++i)
    {
        std::cout << "Face " << i << " neighbors: ";
        for (int neighbor : face_adjacency_list[i])
        {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
    }

    // Plot the mesh
    // igl::opengl::glfw::Viewer viewer;
    // viewer.data().set_mesh(V, F);
    // viewer.launch();

    return 0;
}
