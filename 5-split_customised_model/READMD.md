# Split Customised Model with RPC

This project demonstrates how to split a custom neural network model and distribute the computation across multiple nodes using PyTorch's Remote Procedure Call (RPC). The model in use is a customised variant of a ResNet, and its layers are divided among different workers (or nodes) to be processed.

## Getting Started

1. Ensure you have `PyTorch` and other dependencies installed.
2. Download the dataset (in this case, MNIST) which will be done automatically when you run the central script.
3. Ensure all scripts (`central.py`, `edge.py`, and the model definition) are in the same directory.

## Running the Model

To run the split model, follow these steps:

1. Start the central server:

```bash
python central.py
```

2. In separate terminals, start the edge workers:

```bash
python edge.py --rank 1
python edge.py --rank 2
```

Note: The ranks determine the worker ID and should be unique for each worker.

## What to Expect

Once the central server and the edge workers are running, the model will begin training on the MNIST dataset. The computations are distributed, with certain layers of the neural network being computed on different nodes. After training, you should see the model's accuracy on the test set printed in the terminal.