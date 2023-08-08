import torch.distributed.rpc as rpc
from my_functions import start_to_middle, middle_to_end


def run_middle():
    # Initialize RPC for the "middle" node
    rpc.init_rpc(
        "middle",
        rank=1,
        world_size=3,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500"
        ),
    )

    # Keep the "middle" node running to listen for RPC requests
    # No explicit tasks are performed here, it simply acts as a bridge between the "start" and "end" nodes

    # Shutdown the RPC framework
    rpc.shutdown()


if __name__ == "__main__":
    # Start the RPC "middle" node
    run_middle()
