# server.py
import torch
import torch.distributed.rpc as rpc
from my_function import accumulate_tensors


def run_server(rank):
    rpc.init_rpc(
        f"server{rank}",
        rank=rank,
        world_size=3,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500"
        ),
    )

    x = torch.ones(3, 3) * rank
    fut = rpc.rpc_async("worker", accumulate_tensors, args=(x,))
    result = fut.wait()
    print(f"Result from worker: {result}")

    rpc.shutdown()


if __name__ == "__main__":
    import sys

    run_server(int(sys.argv[1]))
