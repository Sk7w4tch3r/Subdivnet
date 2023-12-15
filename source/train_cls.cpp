#include <torch/torch.h>
#include <torch/script.h>
#include <torch/data/dataloader.h>

#include "subdivnet/network.h"
#include "subdivnet/dataset.h"
#include "subdivnet/utils.h"


std::pair<float, float> train(
    MeshNet & net,
    int const& epoch,
    int const& num_workers,
    torch::Device const& device,
    int const& batch_size = 32){

    auto train_dataset = ClassificationDGGSDataset(
        "data/MNISTDGGS",
        batch_size,
        true,
        true,
        true,
        givde::Resolution(5)
    ).map(Stack<DGGSExample>());

    // net.ptr()->to(device);
    net->to(device);

    net->train();

    int batch_idx = 0;

    torch::optim::Adam adamOptim(net->parameters(), torch::optim::AdamOptions(0.001));
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), torch::data::DataLoaderOptions().workers(num_workers).batch_size(batch_size));
    torch::nn::CrossEntropyLoss cross_entropy = torch::nn::CrossEntropyLoss();


    for (auto& batch : *train_loader){
        auto faces = batch.faces; // not correct
        auto features = batch.feats;
        auto labels = batch.labels;
        
        auto data = features.to(device);
        auto target = labels.to(device);

        MeshTensor mesh_tensor = toMeshTensor(faces, features);
        std::cout << mesh_tensor.shape() << std::endl;


        adamOptim.zero_grad();
        auto output = net->forward(mesh_tensor);
        std::cout << "output sizes " << output.sizes() << std::endl;
        auto loss = cross_entropy(output, target);
        loss.backward();
        adamOptim.step();
        int dataset_size = *train_dataset.size();
        if (batch_idx % 10 == 0){
            std::cout << "Train Epoch: " << epoch << " [" << batch_idx * batch_size << "/" << dataset_size << " (" << 100. * batch_idx / batch_size << "%)]\tLoss: " << 0 << std::endl;
        }
        batch_idx++;
    }

    // net->eval();
    // float test_loss = 0.0;
    // int correct = 0;
    // int bna = 0; // batch number accumulator
    // for (auto& batch : *train_loader){
    //     auto data = batch[bna].data.to(device);
    //     auto target = batch[bna].target.to(device);

    //     MeshTensor mesh_tensor = toMeshTensor(data[0], data[1]);

    //     auto output = net->forward(mesh_tensor);
    //     test_loss += torch::nll_loss(output, target, {}, torch::Reduction::Sum).item<float>();
    //     auto pred = output.argmax(1);
    //     correct += pred.eq(target).sum().item<int>();
    //     bna++;
    // }

    // test_loss /= *train_dataset->size();
    // float acc = 100. * correct / *train_dataset->size();

    return std::make_pair(0.1, 0.1);
}


int main(){
    
    std::string dataroot = "data/MNISTDGGS";
    std::string mode = "train";

    int batch_size = 1;
    int num_workers = 20;
    bool shuffle = true;
    bool in_memory = true;
    int epochs = 10;

    // ClassificationDataset train_dataset = ClassificationDataset(
    //     dataroot,
    //     batch_size,
    //     true,
    //     true,
    //     true,
    //     true
    // );

    auto train_dataset = ClassificationDGGSDataset(
        dataroot,
        batch_size,
        true,
        shuffle,
        in_memory,
        givde::Resolution(5)
    ).map(Stack<DGGSExample>());

    MeshNet net{
        3,
        10,
        4,
        std::vector<int>{32, 64, 128, 128, 128}
    };

    
    float best_acc = 0.0;
    float best_loss = 0;

    float loss = 0.0;
    float acc = 0.0;

    // start training
    if (mode == "train"){
        for (int epoch = 1; epoch <= epochs; ++epoch){
            std::pair<float, float> res = train(net,
                                                epoch, 
                                                num_workers, 
                                                torch::Device(torch::kCUDA),
                                                batch_size);
            loss = res.first;
            acc = res.second;
            if (acc > best_acc){
                best_acc = acc;
            }
            if (loss < best_loss){
                best_loss = loss;
                // get current timestamp
                std::time_t result = std::time(nullptr);
                torch::save(net, "models/" + std::to_string(result) + ".pt");
            }
            
            std::cout << "Epoch: " << epoch << " Loss: " << loss << " Acc: " << acc << std::endl;
        }
    }

    return 0;
}