# Split pretrained model

> Notice that, MNIST and the pretrained resnet18 model different a lot, the transfer learning result is bad, but enough to demonstrate the idea useing RPC.
## Runnable

```bash
python central.py

python edge.py --rank 1
python edge.py --rank 2

```