#include <torch/torch.h>
#include <torch/script.h>
#include <torch/data/dataloader.h>

#include "subdivnet/network.h"
#include "subdivnet/dataset.h"
#include "subdivnet/utils.h"


std::pair<float, float> train(
    MeshNet net,
    ClassificationDataset* train_dataset,
    int epoch,
    int num_workers,
    torch::Device device){
    
    net->to(device);

    net->train();
    int batch_idx = 0;
    
    torch::optim::Adam adamOptim(net->parameters(), torch::optim::AdamOptions(0.001));

    auto train_loader = torch::data::make_data_loader(
        *train_dataset,
        torch::data::DataLoaderOptions().batch_size(train_dataset->k_batch_size).workers(num_workers)
    );
    torch::nn::CrossEntropyLoss cross_entropy = torch::nn::CrossEntropyLoss();

    for (auto& batch : *train_loader){
        auto data = batch[batch_idx].data;
        auto target = batch[batch_idx].target;
        
        data = data.to(device);
        target = target.to(device);

        MeshTensor mesh_tensor = toMeshTensor(data[0], data[1]);

        adamOptim.zero_grad();
        auto output = net->forward(mesh_tensor);
        auto loss = cross_entropy(output, target);
        loss.backward();
        adamOptim.step();
        if (batch_idx % 10 == 0){
            std::cout << "Train Epoch: " << epoch << " [" << batch_idx * train_dataset->k_batch_size << "/" << *train_dataset->size() << " (" << 100. * batch_idx / train_dataset->k_batch_size << "%)]\tLoss: " << loss.item<float>() << std::endl;
        }
        batch_idx++;
    }

    net->eval();
    float test_loss = 0.0;
    int correct = 0;
    int bna = 0; // batch number accumulator
    for (auto& batch : *train_loader){
        auto data = batch[bna].data.to(device);
        auto target = batch[bna].target.to(device);

        MeshTensor mesh_tensor = toMeshTensor(data[0], data[1]);

        auto output = net->forward(mesh_tensor);
        test_loss += torch::nll_loss(output, target, {}, torch::Reduction::Sum).item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(target).sum().item<int>();
        bna++;
    }

    test_loss /= *train_dataset->size();
    float acc = 100. * correct / *train_dataset->size();

    return std::make_pair(test_loss, acc);
}


int main(){
    
    std::string dataroot = "data";
    std::string mode = "train";

    int batch_size = 32;
    int num_workers = 0;
    bool shuffle = true;
    bool augment = true;
    int epochs = 10;

    ClassificationDataset train_dataset = ClassificationDataset(
        dataroot,
        batch_size,
        true,
        true,
        true,
        true
    );

    MeshNet net = MeshNet();

    
    float best_acc = 0.0;
    float best_loss = 100000.0;

    float loss = 0.0;
    float acc = 0.0;

    // start training
    if (mode == "train"){
        for (int epoch = 1; epoch <= epochs; ++epoch){
            std::pair<float, float> res = train(net,
                                                &train_dataset,
                                                epoch, 
                                                num_workers, 
                                                torch::Device(torch::kCUDA));
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