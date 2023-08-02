# worker.py
import torch
import torch.distributed.rpc as rpc


def run_worker(name, master_addr, rank, world_size):
    """
    The function to run on the worker.

    :param name: The name of this worker
    :param master_addr: The address of the master node
    :param rank: The rank of this worker
    :param world_size: The total number of nodes
    """
    options = {
        "rpc_backend_options": rpc.TensorPipeRpcBackendOptions(
            init_method=f"tcp://{master_addr}"
        )
    }

    # Initialize the RPC framework and set the current worker name, rank and the total number of workers
    rpc.init_rpc(name, rank=rank, world_size=world_size, **options)

    # Wait until RPC finishes
    rpc.shutdown()


if __name__ == "__main__":
    run_worker("worker", "127.0.0.1:8080", 1, 2)
