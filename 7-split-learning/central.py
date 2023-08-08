import torch
from model import SplitModel
import torch.multiprocessing as mp
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm


def run_trainer():
    batch = 5
    ntoken = 10
    ninp = 2
    nhid = 3
    nindices = 3
    nlayers = 4
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )
    model = SplitModel('edge0').cuda()
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load train set
    trainset = torchvision.datasets.MNIST(
        root="../data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    # Load test set
    testset = torchvision.datasets.MNIST(
        root="../data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )
    
    
    # setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )
    criterion = torch.nn.CrossEntropyLoss()
    
     # Specify the number of epochs
    epochs = 10

    for epoch in range(epochs):
        for i, data in tqdm(enumerate(trainloader, 0)):
            with dist_autograd.context() as context_id:
                inputs, labels = data
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                model.to("cuda")
                output = model(inputs)
                loss = criterion(output, labels)
                # run distributed backward pass
                dist_autograd.backward(context_id, [loss])
                # run distributed optimizer
                opt.step(context_id)
                # not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        # print statistics
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, loss.item()))

    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )
    
    # block until all rpcs finish
    rpc.shutdown()
        
def run_worker(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '29500'
    print("Running central...")
    options = rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:29500")
    options.set_device_map("edge0", {0: 0})
    #options.set_device_map("central", {0: 0})
    # device_map = {"edge1": {"cuda:0": "cuda:0"}}
    # backend_opts = rpc.TensorPipeRpcBackendOptions(
    #     device_maps=device_map,
    # )
    #options = rpc.TensorPipeRpcBackendOptions()
    #options.set_device_map("edge1", {0: 0})  # Maps central's GPU 0 to edge1's GPU 0
    #options.set_device_map("edge2", {0: 0})  # Maps central's GPU 0 to edge2's GPU 0
    rpc.init_rpc("central", 
                 rank=rank, 
                 world_size=world_size,  
                 rpc_backend_options=options)
    
    
if __name__=="__main__":
    world_size = 2
    rank = 1
    run_worker(rank, world_size)
    run_trainer()