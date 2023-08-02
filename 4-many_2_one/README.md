# Many to one

## Runable

```bash
python worker.py

python server.py 1
python server.py 2
```

## error report, need to use async not sync

Indeed, in a many-to-one scenario where the worker continues running indefinitely to handle requests, it's better to use rpc_async to avoid blocking the server nodes. The worker will process requests as they come in, so there's no need to wait for a response synchronously.

## Tips:

- Use async instead of sync is enough.