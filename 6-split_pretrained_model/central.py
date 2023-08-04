import os
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


from tqdm import tqdm

# Initialize the model
# Load pretrained ResNet18 model
model = models.resnet18(pretrained=True)

# Change the input layer
model.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)

# Change the output layer
model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)

model = nn.ModuleList(
    [
        nn.Sequential(*list(model.children())[:5]),
        nn.Sequential(*list(model.children())[5:6]),
        nn.Sequential(*list(model.children())[6:7]),
        nn.Sequential(*list(model.children())[7:9]),
        nn.Sequential(nn.Flatten(), *list(model.children())[9:]),
    ]
)

# Train all layers in central nodes
for param in model.parameters():
    param.requires_grad = True

model = model.cuda()


def run_server():
    options = rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:8080")
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
        trainset, batch_size=128, shuffle=True, num_workers=16
    )

    # Load test set
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=16
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Create the checkpoints directory if it doesn't exist
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        try:
            os.makedirs(checkpoints_dir)
        except OSError as e:
            print(f"Error creating directory: {checkpoints_dir} : {str(e)}")
            return

    # Load checkpoint if it exists
    model_name = "resnet"
    uid = "01010"  # Change this to your actual UID
    checkpoint_path = f"checkpoints/{model_name}_{uid}.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from epoch {start_epoch}")

    # Specify the number of epochs
    epochs = start_epoch + 5

    for epoch in range(start_epoch, epochs):
        losses = []
        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to("cuda"), labels.to("cuda")

            # Forward through first layer
            x = model[0](inputs)

            # Forward through second layer (edge node 1)
            x = rpc.rpc_sync("edge1", model[1], args=(x,))

            # Forward through third layer
            x = model[2](x)

            # Forward through fourth layer (edge node 2)
            x = rpc.rpc_sync("edge2", model[3], args=(x,))

            # Forward through fifth layer
            output = model[4](x)

            # Compute loss
            loss = loss_fn(output, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # print statistics
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, sum(losses) / len(losses)))

        # Save checkpoint at the end of each epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint for epoch {epoch + 1}")

    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to("cuda"), labels.to("cuda")

            x = model[0](images)
            x = rpc.rpc_sync("edge1", model[1], args=(x,))
            x = model[2](x)
            x = rpc.rpc_sync("edge2", model[3], args=(x,))
            outputs = model[4](x)

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
