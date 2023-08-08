# Many to One RPC Communication

In this setup, multiple servers send remote procedure call (RPC) requests to a single worker. The worker processes requests asynchronously, ensuring that it can handle multiple requests without blocking any individual server.

## Getting Started

### Prerequisites

- PyTorch and its distributed RPC module installed.

### Running the Code

To start the worker and servers, follow the commands below:

```bash
# Start the worker first
python worker.py

# Then start the servers
python server.py 1
python server.py 2
```

The worker will await and process RPC requests from both servers.

## Important Notes

1. **Asynchronous Processing**: In a many-to-one scenario like this, it's essential to use asynchronous RPCs (`rpc_async`) instead of synchronous ones (`rpc_sync`). This change ensures the worker doesn't block any server and can handle multiple requests concurrently.

2. **Error Handling**: If you encounter issues with synchronous calls, the solution is to switch to asynchronous calls. In the provided setup, `rpc_async` is recommended over `rpc_sync`.

## Tips

- Always use `rpc_async` instead of `rpc_sync` in many-to-one RPC scenarios to avoid potential deadlocks and to ensure smooth processing.
