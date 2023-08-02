# server.py
import torch
import torch.distributed.rpc as rpc
from my_function import add_tensors


def run_server(master_addr, world_size):
    """
    The function to run on the server.

    :param master_addr: The address of the master node
    :param world_size: The total number of nodes
    """
    options = {
        "rpc_backend_options": rpc.TensorPipeRpcBackendOptions(
            init_method=f"tcp://{master_addr}",
        )
    }

    # Initialize the RPC framework with the name 'server', rank 0 and the total number of nodes
    rpc.init_rpc("server", rank=0, world_size=world_size, **options)

    # Create two tensors
    x = torch.ones(3, 3)
    y = torch.ones(3, 3)

    # Call the function on each of the workers
    for i in range(1, world_size):
        result = rpc.rpc_sync(f"worker{i}", add_tensors, args=(x, y))
        x = result

        # Print the result
        print(f"Result from worker{i}: {result}")

    # Shutdown the RPC framework
    rpc.shutdown()


if __name__ == "__main__":
    run_server("127.0.0.1:8080", 3)
