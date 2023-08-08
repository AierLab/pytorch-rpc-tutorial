import torch
import torch.distributed.rpc as rpc
import argparse
import os
from model import SplitModel
#from model import SimpleResnet

# Initialize the model


def run_worker(rank, world_size):
    
    print(f"Running edge{rank}...")
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:29500")
    rpc.init_rpc(
        f"edge{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0, help="Rank of this worker")
    parser.add_argument(
        "--world_size", type=int, default=2, help="Total number of workers"
    )
    args = parser.parse_args()

    run_worker(args.rank, args.world_size)