# server.py
import torch
import torch.distributed.rpc as rpc
from my_function import add_tensors


def run_server(name, worker_addr, world_size):
    """
    The function to run on the server.

    :param name: The name of the server
    :param worker_addr: The address of the worker node
    :param world_size: The total number of nodes
    """
    options = {
        "rpc_backend_options": rpc.TensorPipeRpcBackendOptions(
            init_method=f"tcp://{worker_addr}"
        )
    }

    # Initialize the RPC framework and set the current worker name, rank and the total number of workers
    rpc.init_rpc(name, rank=0, world_size=world_size, **options)

    # Create two tensors
    x = torch.ones(3, 3)
    y = torch.ones(3, 3)

    # Call the function on the worker
    result = rpc.rpc_sync("worker", add_tensors, args=(x, y))

    # Print the result
    print(result)

    # Shutdown the RPC framework
    rpc.shutdown()


if __name__ == "__main__":
    run_server("server", "127.0.0.1:8080", 2)
