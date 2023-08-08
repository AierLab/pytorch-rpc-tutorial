import os
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

# --- MODEL SETUP ---
# Load the pretrained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the input layer to take single channel (grayscale) images
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)

# Modify the output layer to have 10 output classes
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)

# Split the model into different segments
model = nn.ModuleList(
    [
        nn.Sequential(*list(model.children())[:5]),  # First segment
        nn.Sequential(*list(model.children())[5:6]),  # Second segment
        nn.Sequential(*list(model.children())[6:7]),  # Third segment
        nn.Sequential(*list(model.children())[7:9]),  # Fourth segment
        nn.Sequential(nn.Flatten(), *list(model.children())[9:]),  # Fifth segment
    ]
)

# Ensure all layers are trainable on central nodes
for param in model.parameters():
    param.requires_grad = True

# Move the model to the GPU
model = model.cuda()


def run_server():
    # --- RPC SETUP ---
    options = rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:8080")
    # Setting up device map for edge nodes
    options.set_device_map("edge1", {0: 0})  # Maps central's GPU 0 to edge1's GPU 0
    options.set_device_map("edge2", {0: 0})  # Maps central's GPU 0 to edge2's GPU 0
    rpc.init_rpc("central", rank=0, world_size=3, rpc_backend_options=options)

    # --- DATA LOADING ---
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=16
    )

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=16
    )

    # --- OPTIMIZER AND LOSS ---
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # --- CHECKPOINT HANDLING ---
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    model_name = "resnet"
    uid = "01010"  # Replace with a unique identifier
    checkpoint_path = f"checkpoints/{model_name}_{uid}.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from epoch {start_epoch}")

    # --- TRAINING LOOP ---
    epochs = start_epoch + 5
    for epoch in range(start_epoch, epochs):
        losses = []
        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # Forward pass through segments of the model
            x = model[0](inputs)
            x = rpc.rpc_sync("edge1", model[1], args=(x,))
            x = model[2](x)
            x = rpc.rpc_sync("edge2", model[3], args=(x,))
            output = model[4](x)

            # Compute loss and backpropagate
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Logging and saving
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, sum(losses) / len(losses)))
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint for epoch {epoch + 1}")

    # --- MODEL EVALUATION ---
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            # Forward pass through segments of the model
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
