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
    Adds two tensors together and then forwards the result to the 'end' node
    to further process the result.

    :param x: A PyTorch tensor
    :param y: Another PyTorch tensor
    :return: The processed result from the 'end' node
    """

    # Add the two tensors together
    z = torch.add(x, y)

    # Print the intermediate result
    print(f"Intermediate result at middle: {z}")

    # Make a synchronous RPC call to the 'end' node for further processing
    result = rpc.rpc_sync("end", middle_to_end, args=(z,))

    # Return the final result
    return result
