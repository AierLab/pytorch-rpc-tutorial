import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from model import SimpleResnet

# Load and initialize the SimpleResnet model
model = SimpleResnet().cuda()

# Load pretrained weights
# Uncomment the next line if you have pretrained weights
# model.load_state_dict(torch.load("resnet_pretrained.pth"))


def run_server():
    # Initialize RPC with options
    options = rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:29500")
    # Map central's GPU 0 to the GPUs of edge1 and edge2
    options.set_device_map("edge1", {0: 0})
    options.set_device_map("edge2", {0: 0})

    rpc.init_rpc("central", rank=0, world_size=3, rpc_backend_options=options)

    # Data transformation for normalization
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load MNIST training dataset
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    # Load MNIST test dataset
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU

            # Split model layers and use RPC for remote layers
            x = model.layers[0](inputs)
            x = rpc.rpc_sync("edge1", model.layers[1], args=(x,))
            x = model.layers[2](x)
            x = rpc.rpc_sync("edge2", model.layers[3], args=(x,))
            output = model.layers[4:](x)

            # Compute loss, backpropagate, and update weights
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, loss.item()))

    # Evaluation on the test set
    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            # Similar layer split and RPC calls as training
            x = model.layers[0](images)
            x = rpc.rpc_sync("edge1", model.layers[1], args=(x,))
            x = model.layers[2](x)
            x = rpc.rpc_sync("edge2", model.layers[3], args=(x,))
            outputs = model.layers[4:](x)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy on test images: %d %%" % (100 * correct / total))

    rpc.shutdown()


if __name__ == "__main__":
    run_server()
