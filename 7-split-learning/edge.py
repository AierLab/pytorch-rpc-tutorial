import torch
import torch.distributed.rpc as rpc
import argparse
import os
from model import SplitModel

# If you later need to use another model, you can import it like:
# from model import SimpleResnet


def run_worker(rank: int, world_size: int):
    """
    Initialize the RPC framework and set up the worker.

    Args:
        rank (int): Unique identifier for the worker. Also determines its role (server/client).
        world_size (int): Total number of workers participating in the distributed setup.
    """
    print(f"Running edge{rank}...")

    # TensorPipe is a transport library that enhances PyTorch's capabilities in distributed scenarios.
    # Here, we're setting the initialization method to use TCP and bind to localhost on port 8080.
    options = rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:29500")

    # Initializing the RPC framework.
    # It allows each worker to communicate and provides methods for remote method invocation.
    rpc.init_rpc(
        name=f"edge{rank}",  # Giving each worker a unique name based on its rank.
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options,
    )

    # Shutdown RPC to free resources.
    rpc.shutdown()


if __name__ == "__main__":
    # Argument parser setup to facilitate command line arguments input when running the script.
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=1, help="Rank of this worker")
    parser.add_argument(
        "--world_size", type=int, default=2, help="Total number of workers"
    )

    # Parse the command line arguments.
    args = parser.parse_args()

    # Start the worker with the provided rank and world_size.
    run_worker(args.rank, args.world_size)
