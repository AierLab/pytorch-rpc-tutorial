# end.py

import torch.distributed.rpc as rpc
from my_functions import middle_to_end


def run_end():
    rpc.init_rpc(
        "end",
        rank=2,
        world_size=3,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500"
        ),
    )

    rpc.shutdown()


if __name__ == "__main__":
    run_end()
