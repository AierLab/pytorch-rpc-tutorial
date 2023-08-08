import torch
import torch.distributed.rpc as rpc


def run_worker():
    """
    Initialize and run the RPC worker.
    The worker waits for incoming RPCs, executes them, and sends back the results.
    """

    # Initialize the RPC for this worker
    rpc.init_rpc(
        "worker",
        rank=0,
        world_size=3,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500"
        ),
    )

    # Note: The worker will remain alive and await RPC calls until it's manually shut down
    # In this script, the worker is immediately shut down for demonstration purposes

    # Shutdown the RPC for this worker
    rpc.shutdown()


if __name__ == "__main__":
    run_worker()
