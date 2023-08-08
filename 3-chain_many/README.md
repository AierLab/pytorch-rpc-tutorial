# Chain Many - Distributed RPC with PyTorch

This application demonstrates a chained RPC call in a distributed setting using PyTorch's RPC framework. The nodes involved in the chain are `start`, `middle`, and `end`. 

## Execution

To run the application, execute the following commands in separate terminals:

```bash
python start.py
python middle.py
python end.py
```

## How it works

1. `start` node initiates an RPC call (`rpc_sync`) to `middle`.
2. `middle` node processes the incoming data from `start`. During this processing, `middle` makes another RPC call (`rpc_sync`) to `end`.
3. `end` node processes the data sent by `middle` and returns a result.
4. `middle` receives the result from `end`, possibly does further processing, and sends a result back to `start`.
5. Finally, the `start` node receives the result from its initial RPC call to `middle`.

### Key Insights

- If there's a need to make a call to another node in the middle of an `rpc_sync`, adding another `rpc_sync` within the function is indeed the way to proceed.
- Although `middle.py` and `end.py` are structured similarly in this example, representing nodes waiting for incoming RPCs, in more complex settings they might be substantially different based on their respective roles or resources.
- Excessive use of `rpc_sync` might lead to performance bottlenecks due to its blocking nature. Where possible, consider `rpc_async` for non-blocking calls.
