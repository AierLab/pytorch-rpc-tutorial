import torch
import torch.distributed.rpc as rpc
from my_functions import start_to_middle


def run_start():
    # Initialize RPC for the "start" node
    rpc.init_rpc(
        "start",
        rank=0,
        world_size=3,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500"
        ),
    )

    # Create two sample tensors for demonstration
    x = torch.ones(3, 3)
    y = torch.ones(3, 3)

    # Make a synchronous RPC call to the "middle" node to process the tensors
    result = rpc.rpc_sync("middle", start_to_middle, args=(x, y))

    # Print the result received from the "middle" node
    print(f"Result from end via middle: {result}")

    # Shutdown the RPC framework
    rpc.shutdown()


if __name__ == "__main__":
    # Start the RPC "start" node
    run_start()
