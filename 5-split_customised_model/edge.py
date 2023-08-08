import torch
import torch.distributed.rpc as rpc
import argparse
from model import SimpleResnet

# Initialize the model
model = SimpleResnet().cuda()


def run_worker(rank, world_size):
    """
    Initialize the RPC for the worker with the given rank and shutdown after initialization.

    Args:
    - rank (int): Rank of this worker.
    - world_size (int): Total number of workers.
    """
    rpc.init_rpc(
        f"edge{rank}",  # Name of the RPC for this worker
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500"
        ),
    )

    # Shutdown the RPC for this worker
    rpc.shutdown()


if __name__ == "__main__":
    # Argument parser to receive command line arguments
    parser = argparse.ArgumentParser(description="Run RPC worker.")
    parser.add_argument("--rank", type=int, default=0, help="Rank of this worker.")
    parser.add_argument(
        "--world_size", type=int, default=3, help="Total number of workers."
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Run the worker with the given arguments
    run_worker(args.rank, args.world_size)
