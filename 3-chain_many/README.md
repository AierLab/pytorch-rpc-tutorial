# Chain many

## Runable
```bash
python start.py
python middle.py
python end.py
```
## If we want call another node to do something in the middle of a rpc_sync, it's better to update the function call, add another rpc_sync, am I correct?

Yes, you're correct. If you need to make a call to another node within the `rpc_sync` call, you can certainly do so by adding another `rpc_sync` (or `rpc_async` if you don't immediately need the result) within the function that is called by the initial `rpc_sync`.

Here's an example:

Let's assume we have three nodes: `start`, `middle`, and `end`. The `start` node makes an `rpc_sync` call to the `middle` node, and within the function called by this `rpc_sync`, the `middle` node makes another `rpc_sync` call to the `end` node. Here's what this might look like:

```python
# On start node
def run_start():
    # ...
    result = rpc.rpc_sync("middle", middle_function, args=(x, y))
    # ...

# On middle node
def middle_function(x, y):
    # Process x and y...
    # Now need to call a function on the end node
    result = rpc.rpc_sync("end", end_function, args=(z, w))
    # Continue processing...
    return final_result

# On end node
def end_function(z, w):
    # Process z and w...
    return final_result
```

In this example, `start` calls `middle_function` on `middle` with `rpc_sync`. Inside `middle_function`, another `rpc_sync` is used to call `end_function` on `end`.

Remember, the order of operations here matters since `rpc_sync` is blocking. The `middle_function` won't continue to execute until the `rpc_sync` call to the `end` node has completed and returned a result.

Please note that excessive use of `rpc_sync` could potentially lead to performance bottlenecks due to its synchronous nature. If your application can handle more asynchrony, it may be beneficial to use `rpc_async` where it's feasible.


## middle.py and end.py are the same, right?

Yes, that's correct. In this particular example, both `middle.py` and `end.py` are servers that wait for incoming RPCs, execute the received function call, and return the results. 

`middle.py` waits for RPCs from `start.py`, processes the input, then makes RPC calls to `end.py`. `end.py` waits for RPCs from `middle.py`, processes the input, and returns the result. 

The key difference is in the functions they're expected to execute. `middle.py` uses the `start_to_middle` function from `my_functions.py`, and `end.py` uses the `middle_to_end` function from `my_functions.py`. 

The structure of the two files is similar because they both act as RPC servers. However, in a more complex application, the middle and end nodes might perform different operations and have different code. 

Also note that in an actual distributed system, these nodes might be located on different machines, and the tasks they perform could be determined by their specific roles or resources in the overall system.