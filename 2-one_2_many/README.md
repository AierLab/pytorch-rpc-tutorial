# One-to-Many RPC Demo with PyTorch

This application demonstrates a one-to-many RPC communication using PyTorch's RPC framework. The setup consists of one server that can distribute tasks to and communicate with multiple workers.

## How to Run

1. Navigate to the server directory and start the server:

```bash
cd server
python server.py
```

2. In separate terminals, navigate to the worker directory and start each worker:

```bash
cd worker
python worker.py worker1 1
python worker.py worker2 2
```

(Note: The arguments to `worker.py` specify the worker's name and rank.)

## Architecture Overview

### Server
The server acts as a central coordinator, responsible for task distribution and result aggregation. In this demo, it awaits RPC calls from the workers.

### Workers
Each worker is an independent unit that can receive tasks from the server, process them, and send the results back. The workers can be scaled as per the requirement, and each can be identified uniquely by its name and rank.

## Design Insights

- In traditional systems, a one-to-many master-slave (or server-worker) model is common. Here, one central server coordinates multiple workers.
  
- With PyTorch's RPC framework, however, it is possible to have multiple servers and workers, enabling peer-to-peer communication. This means any node can send an RPC to any other node, creating versatile architectures.
  
- While each node in PyTorch's RPC system can act as both a server and a worker, simpler designs or learning demos often feature a single server coordinating multiple workers.

## Conclusion

This demo showcases the flexibility of PyTorch's RPC framework in supporting both traditional and peer-to-peer distributed architectures. Depending on the specific use case and requirements, developers can choose to implement a simple master-slave model or more complex peer-to-peer interactions.
