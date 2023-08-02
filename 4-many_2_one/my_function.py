# my_functions.py

import time
import torch


def accumulate_tensors(x):
    """
    Sums all elements of a tensor

    :param x: A PyTorch tensor
    :return: The sum of all elements in x
    """
    return torch.sum(x)
