# my_functions.py

import torch
import torch.distributed.rpc as rpc


def middle_to_end(x):
    """
    Doubles the input tensor and returns the result.

    :param x: A PyTorch tensor
    :return: The result of doubling x
    """
    return 2 * x


def start_to_middle(x, y):
    """
    Adds two tensors and returns the result.

    :param x: A PyTorch tensor
    :param y: Another PyTorch tensor
    :return: The result of adding x and y
    """
    z = torch.add(x, y)
    print(f"Result from end via middle: {z}")

    # Call the 'end' node asynchronously
    result = rpc.rpc_sync("end", middle_to_end, args=(z,))

    return result
