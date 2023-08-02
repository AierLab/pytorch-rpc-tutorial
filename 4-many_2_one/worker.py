# worker.py

import torch
import torch.distributed.rpc as rpc


def run_worker():
    rpc.init_rpc(
        "worker",
        rank=0,
        world_size=3,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500"
        ),
    )

    rpc.shutdown()


if __name__ == "__main__":
    run_worker()
