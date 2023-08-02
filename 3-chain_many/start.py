# start.py

import torch
import torch.distributed.rpc as rpc
from my_functions import start_to_middle


def run_start():
    rpc.init_rpc(
        "start",
        rank=0,
        world_size=3,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500"
        ),
    )

    # Create two tensors
    x = torch.ones(3, 3)
    y = torch.ones(3, 3)

    # Call the 'middle' node asynchronously
    result = rpc.rpc_sync("middle", start_to_middle, args=(x, y))
    print(f"Result from end via middle: {result}")

    rpc.shutdown()


if __name__ == "__main__":
    run_start()
