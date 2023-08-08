# Split Pretrained Model Tutorial

In this tutorial, we'll learn how to split a pretrained model using RPC (Remote Procedure Call). We'll use the MNIST dataset and a pretrained `resnet18` model. While these two don't naturally align well (and the transfer learning result might not be optimal), it serves to demonstrate the core idea behind using RPC.

> 🚩 **Note**: The MNIST dataset and the pretrained `resnet18` model differ significantly. Hence, while using this pretrained model for MNIST might result in suboptimal performance, the purpose of this exercise is to showcase the RPC technique.

## Setup

1. Make sure you have all the necessary libraries installed.
2. Ensure you have both `central.py` and `edge.py` scripts in your working directory.

## Execution

Follow the steps below to run the tutorial:

1. Start the central node:
```bash
python central.py
```

2. In separate terminals, start two edge nodes:
```bash
python edge.py --rank 1
```
```bash
python edge.py --rank 2
```

Wait for the processes to complete, and then you can analyze the results.
