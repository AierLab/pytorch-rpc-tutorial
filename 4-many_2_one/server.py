import torch
import torch.distributed.rpc as rpc
from my_function import accumulate_tensors


def run_server(rank):
    """
    Initialize the RPC server, send a tensor to a worker, and retrieve the results.

    Args:
    - rank (int): The rank of this server. Determines the server's identity.
    """

    # Initialize the RPC for this server
    rpc.init_rpc(
        f"server{rank}",
        rank=rank,
        world_size=3,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500"
        ),
    )

    # Create a tensor with values multiplied by the rank
    x = torch.ones(3, 3) * rank

    # Asynchronously send the tensor to the worker for processing
    fut = rpc.rpc_async("worker", accumulate_tensors, args=(x,))

    # Wait for the results from the worker
    result = fut.wait()

    # Print the results
    print(f"Result from worker: {result}")

    # Shutdown the RPC for this server
    rpc.shutdown()


if __name__ == "__main__":
    import sys

    # Retrieve the server rank from the command-line arguments and run the server
    run_server(int(sys.argv[1]))
