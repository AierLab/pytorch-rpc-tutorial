import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from model import SimpleResnet

# Initialize the model
model = SimpleResnet().cuda()

# Load pretrained weights
# model.load_state_dict(torch.load("resnet_pretrained.pth"))  # Uncomment this if you have pretrained weights


def run_server():
    options = rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:29500")
    options.set_device_map("edge1", {0: 0})  # Maps central's GPU 0 to edge1's GPU 0
    options.set_device_map("edge2", {0: 0})  # Maps central's GPU 0 to edge2's GPU 0

    rpc.init_rpc("central", rank=0, world_size=3, rpc_backend_options=options)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load train set
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    # Load test set
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Specify the number of epochs
    epochs = 10

    for epoch in range(epochs):
        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to("cuda"), labels.to("cuda")

            # Forward through first layer
            x = model.layers[0](inputs)

            # Forward through second layer (edge node 1)
            x = rpc.rpc_sync("edge1", model.layers[1], args=(x,))

            # Forward through third layer
            x = model.layers[2](x)

            # Forward through fourth layer (edge node 2)
            x = rpc.rpc_sync("edge2", model.layers[3], args=(x,))

            # Forward through fifth layer
            output = model.layers[4:](x)

            # Compute loss
            loss = loss_fn(output, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print statistics
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, loss.item()))

    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to("cuda"), labels.to("cuda")

            x = model.layers[0](images)
            x = rpc.rpc_sync("edge1", model.layers[1], args=(x,))
            x = model.layers[2](x)
            x = rpc.rpc_sync("edge2", model.layers[3], args=(x,))
            outputs = model.layers[4:](x)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )

    rpc.shutdown()


if __name__ == "__main__":
    run_server()
