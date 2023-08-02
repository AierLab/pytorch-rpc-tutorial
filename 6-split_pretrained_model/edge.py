import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torchvision.models as models

import argparse


# Initialize the model
# Load pretrained ResNet18 model
model = models.resnet18(pretrained=True)

# Change the input layer
model.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)

# Change the output layer
model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)

# Optionally, if you want to fine-tune only the modified layers
for param in model.parameters():
    param.requires_grad = False

for param in model.conv1.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

model = model.cuda()
model = nn.ModuleList(
    [
        nn.Sequential(*list(model.children())[:5]),
        nn.Sequential(*list(model.children())[5:6]),
        nn.Sequential(*list(model.children())[6:7]),
        nn.Sequential(*list(model.children())[7:8]),
        nn.Sequential(*list(model.children())[8:]),
    ]
)


def run_worker(rank, world_size):
    rpc.init_rpc(
        f"edge{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:8888"
        ),
    )

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0, help="Rank of this worker")
    parser.add_argument(
        "--world_size", type=int, default=3, help="Total number of workers"
    )
    args = parser.parse_args()

    run_worker(args.rank, args.world_size)
