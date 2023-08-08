# Split Learning with PyTorch Distributed RPC

In split learning, we divide a neural network into multiple segments and run different segments on various devices. This tutorial will demonstrate how to use PyTorch's Distributed RPC framework to implement split learning.

## Introduction

Split learning is a form of distributed deep learning where the model is split across devices, and each device processes an individual segment of the model. This type of setup is especially beneficial in cases where data cannot leave the local device due to privacy or bandwidth concerns.

Our example here uses the MNIST dataset and a convolutional neural network (CNN) split across two devices: a central node and an edge device.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- tqdm

## Setup

First, ensure you have all required libraries installed. You can use the following command to install the necessary packages:

```bash
pip install torch torchvision tqdm
```

## How to Run

1. Start the central node:

```bash
python central.py
```

2. In a separate terminal, start the edge device:

```bash
python edge.py
```

## Code Structure

- `central.py`: Contains the code for initializing the RPC framework for the central node and running the training loop.
  
- `edge.py`: Initializes the edge device's RPC setup.

- `model.py`: Defines the split model with parts of it designated for remote execution.

## Explanation

The training loop in `central.py` loads the MNIST dataset and trains the split model for a specified number of epochs. The forward pass of the model executes parts of the model on the edge device remotely using RPC calls. During the backward pass, gradients are computed using distributed autograd and synchronized across the central and edge devices. The central node then updates the model parameters using these synchronized gradients.

The model defined in `model.py` is a simple CNN. Some layers of this network are set up to run on the edge device, as specified by the `SplitModel` class.

## Conclusion

Split learning with PyTorch's Distributed RPC allows for flexible and efficient distributed deep learning setups, especially in scenarios with data privacy concerns. This tutorial provides a foundation to explore more complex scenarios and topologies.
