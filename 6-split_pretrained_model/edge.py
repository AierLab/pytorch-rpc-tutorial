import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torchvision.models as models
import argparse

# --- MODEL INITIALIZATION ---

# Load the pretrained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the input layer to accept grayscale images
model.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False,
)

# Modify the output layer for 10 classes
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

# Set all layers to non-trainable for edge nodes
for param in model.parameters():
    param.requires_grad = False

# Move the model to the GPU
model = model.cuda()


def run_worker(rank, world_size):
    """
    Initializes a worker for RPC communication.
    """

    # Set up RPC for this worker
    rpc.init_rpc(
        name=f"edge{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:8080"
        ),
    )

    # Shutdown RPC for this worker after it's done
    rpc.shutdown()


if __name__ == "__main__":
    # Argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Initialize a RPC worker.")
    parser.add_argument("--rank", type=int, default=0, help="Rank of this worker.")
    parser.add_argument(
        "--world_size", type=int, default=3, help="Total number of workers."
    )

    args = parser.parse_args()

    # Run the worker with given rank and world size
    run_worker(args.rank, args.world_size)
