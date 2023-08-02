# my_functions.py

import torch


def add_tensors(x, y):
    """
    Adds two tensors

    :param x: A PyTorch tensor
    :param y: Another PyTorch tensor
    :return: The result of adding x and y
    """
    return torch.add(x, y)
