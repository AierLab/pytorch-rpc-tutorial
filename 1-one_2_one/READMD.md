# One-to-One RPC Demo with PyTorch

This application demonstrates a one-to-one RPC communication using PyTorch's RPC framework. The setup consists of a server coordinating with a single worker.

## How to Run

1. Navigate to the server directory and start the server:

```bash
cd server
python server.py
```

2. In a separate terminal, navigate to the worker directory and start the worker:

```bash
cd worker
python worker.py
```

## Architecture Overview

### Server
The server acts as a coordinator, initiating RPC calls and orchestrating the entire process. It sends data to the worker for processing and then retrieves the results.

### Worker
The worker receives tasks from the server, processes them, and sends the results back. It is defined to execute specific functions that the server can call remotely.

## Design Insights

### Relationship between Worker and Server
In the context of this demo:

- **Worker:** A computational unit that performs tasks upon receiving RPCs. It processes data and functions as per the remote calls from the server.
  
- **Server:** Acts as the orchestrator, making RPCs to the worker, directing it to execute certain tasks, and gathering the results.

While the server-worker relationship can often be viewed as master-slave, with the server directing operations, it's essential to understand that more complex setups can blur these roles, enabling nodes to function both as servers and workers.

## Tips

- Ensure consistency in project or function structures between the server and worker. This facilitates relative function calls without conflicts.
  
- If a function exists on the worker but isn't needed for computation on the server, you can create a dummy function with the same name on the server. This placeholder ensures compatibility without the need for actual implementation on the server side.
