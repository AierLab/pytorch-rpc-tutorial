# One-to-one rpc demo

## Runable
```bash
cd server
python server.py

cd worker
python worker.py 
```

## What is the relationship between worker and server?

In the context of the PyTorch RPC (Remote Procedure Call) framework, a worker and a server are both nodes in a distributed system that can execute RPCs.

1. **Worker:** In distributed computing, a worker usually refers to a computing unit (a CPU or a GPU, for example) that is responsible for performing tasks assigned by a central authority (like a server or master node). In the given PyTorch RPC code, the worker is defined to perform specific functions (`add_tensors`) that the server can call remotely.

2. **Server:** In the RPC setting, the server (also known as the master node) typically orchestrates the distributed system by assigning tasks to the workers and gathering results from them. The server in the provided code is responsible for initializing the RPC system, creating the data (two tensors), making an RPC to the worker to execute a function on the data, and then retrieving and printing the results.

The relationship between the worker and server is a master-slave relationship. The server can call functions on the worker remotely, get the results, and use these results for further computation. The worker executes the function calls it receives from the server.

Please note that in a more advanced setting, the distinction between workers and servers can blur, with nodes capable of acting as both a server and a worker – sending and receiving RPCs. This could lead to more complex and versatile distributed systems.

## Tips

- The function or the project structure on the server side and client should be the same, in order to call functions relatively.
- You can even use dummy function with no implementation but the same name on the server side, if the server.py never use this function doing real things.