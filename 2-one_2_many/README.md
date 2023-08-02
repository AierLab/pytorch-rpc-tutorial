# One-to-many rpc demo

## Runable
```bash
cd server
python server.py

cd worker
python worker.py worker1 1
python worker.py worker2 2
```

## Can I have multiple worker and only one server?

In a typical master-slave architecture, you would indeed have one master (server) and multiple slaves (workers). The server is responsible for coordinating the work among the workers, distributing tasks, and gathering results. 

However, with PyTorch's RPC framework, you can have multiple servers and multiple workers. This is due to the framework's support for peer-to-peer communication, meaning any node can send a remote procedure call to any other node. This allows for more complex and versatile architectures, beyond just a simple master-slave model. 

In essence, every node in PyTorch's RPC framework can act as both a server and a worker depending on whether it is sending or receiving an RPC. So you could design a system where one node distributes tasks to other nodes (acting as a server), and also does some computation itself when it receives an RPC from another node (acting as a worker). The design of the system really depends on your specific needs and use case. 

But for simpler applications or for learning purposes, it's common to stick with a single server that distributes tasks to multiple workers.