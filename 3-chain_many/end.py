import torch.distributed.rpc as rpc
from my_functions import middle_to_end


def run_end():
    # Initialize RPC for the "end" node
    rpc.init_rpc(
        "end",
        rank=2,
        world_size=3,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500"
        ),
    )

    # The "end" node is set up to listen for RPC requests from the "middle" node.
    # No explicit tasks are performed here, but it's expected to handle incoming requests.

    # Shutdown the RPC framework after all RPC tasks are completed
    rpc.shutdown()


if __name__ == "__main__":
    # Start the RPC "end" node
    run_end()
