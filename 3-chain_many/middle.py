# middle.py

import torch.distributed.rpc as rpc
from my_functions import start_to_middle, middle_to_end


def run_middle():
    rpc.init_rpc(
        "middle",
        rank=1,
        world_size=3,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500"
        ),
    )

    rpc.shutdown()


if __name__ == "__main__":
    run_middle()
