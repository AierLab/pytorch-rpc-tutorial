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
    """
    The main training loop for the distributed model. It loads data,
    runs the training loop, and evaluates the model.
    """

    # Initialize the SplitModel; the first layer will be remotely executed on "edge1" worker.
    model = SplitModel("edge1").cuda()

    # Standard transformations for the MNIST dataset.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load training data from MNIST dataset.
    trainset = torchvision.datasets.MNIST(
        root="../data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    # Load test data from MNIST dataset.
    testset = torchvision.datasets.MNIST(
        root="../data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    # Setup the distributed optimizer; gradients will be synchronized across nodes.
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Training loop for a specified number of epochs.
    epochs = 10
    for epoch in range(epochs):
        for i, data in tqdm(enumerate(trainloader, 0)):
            with dist_autograd.context() as context_id:
                inputs, labels = data
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                model.to("cuda")
                output = model(inputs)
                loss = criterion(output, labels)

                # Compute gradients using distributed autograd context.
                dist_autograd.backward(context_id, [loss])

                # Update model parameters.
                opt.step(context_id)

        # Print training statistics.
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, loss.item()))

    # Evaluate the model on the test set.
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

    # Shutdown the RPC service gracefully.
    rpc.shutdown()


def run_worker(rank: int, world_size: int):
    """
    Initialize the RPC framework and set up the central worker.

    Args:
        rank (int): The rank of this worker, typically 0 for the central node.
        world_size (int): Total number of workers participating in the distributed setup.
    """
    print("Running central...")

    # TensorPipe is a transport library to enhance PyTorch's distributed capabilities.
    options = rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:29500")

    # Mapping the GPU on central to the GPU on edge1.
    options.set_device_map("edge1", {0: 0})

    # Initializing the RPC framework for communication.
    rpc.init_rpc(
        "central", rank=rank, world_size=world_size, rpc_backend_options=options
    )


if __name__ == "__main__":
    world_size = 2
    rank = 0

    # Initialize the RPC worker and start the training loop.
    run_worker(rank, world_size)
    run_trainer()
