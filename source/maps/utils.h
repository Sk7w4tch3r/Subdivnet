#include <torch/torch.h>
#include <vector>

bool check_duplicated(torch::Tensor V) {
        
}

std::vector<int> maximal_independent_set(
    std::vector<int> Vids, 
    torch::Tensor Faces, 
    std::vector<std::vector<int>> vertex_faces){
    
    std::vector<bool> marks(Vids.size(), false);
    std::vector<int> result;

    for (auto v : Vids){
        if (!marks[v]){
            result.push_back(v);
            marks[v] = true;

            for (auto f : vertex_faces[v]){
                for (auto v2 : Faces[f]){
                    marks[v2] = true;
                }
            }
        }
    }

    return result;

}