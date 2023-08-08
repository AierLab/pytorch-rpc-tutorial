import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, remote, rpc_sync

# ----------------- HELPER FUNCTIONS ------------------


def _call_method(method: callable, rref: RRef, *args, **kwargs) -> torch.Tensor:
    """
    Calls a given method on the local value of an RRef.

    Args:
        method (callable): The method to be invoked.
        rref (RRef): The RRef which holds the remote module.
        *args: Variable length argument list for the method.
        **kwargs: Arbitrary keyword arguments for the method.

    Returns:
        torch.Tensor: The result from calling the method.
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method: callable, rref: RRef, *args, **kwargs) -> torch.Tensor:
    """
    Sends an RPC to the owner of the RRef to call a specific method.

    Args:
        method (callable): The method to be invoked.
        rref (RRef): The RRef which holds the remote module.
        *args: Variable length argument list for the method.
        **kwargs: Arbitrary keyword arguments for the method.

    Returns:
        torch.Tensor: The result from calling the method on the remote module.
    """
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _parameter_rrefs(module: nn.Module) -> list:
    """
    Returns RRefs (Remote References) for all parameters of a given module.

    Args:
        module (nn.Module): The PyTorch module.

    Returns:
        list: A list of RRefs for the module's parameters.
    """
    return [RRef(param) for param in module.parameters()]


# ----------------- INDIVIDUAL MODEL LAYERS ------------------


class Layer_0(nn.Module):
    """
    The first layer of the model which is meant to run on a remote worker with GPU support.
    """

    def __init__(self):
        super(Layer_0, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        "Move to GPU (remote)"
        self.layers.to("cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the layer."""
        return self.layers(x)


class Layer_1(nn.Module):
    def __init__(self):
        super(Layer_1, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.layers(x)


class Layer_2(nn.Module):
    def __init__(self):
        super(Layer_2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )
        "Move to GPU (remote)"
        self.layers.to("cuda")

    def forward(self, x):
        return self.layers(x)


class Layer_3(nn.Module):
    def __init__(self):
        super(Layer_3, self).__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)


class SplitModel(nn.Module):
    """
    A model architecture that's split across multiple devices for distributed training using PyTorch's RPC framework.
    """

    def __init__(self, worker: str):
        super(SplitModel, self).__init__()
        self.layer_0_rref = rpc.remote(worker, Layer_0, args=())  # Remote layer
        self.layer_1 = Layer_1()  # Local layer
        self.layer_2_rref = rpc.remote(worker, Layer_2, args=())  # Remote layer
        self.layer_3 = Layer_3()  # Local layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for the split model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        x = _remote_method(Layer_0.forward, self.layer_0_rref, x)
        x = self.layer_1(x)
        x = _remote_method(Layer_2.forward, self.layer_2_rref, x)
        x = self.layer_3(x)
        return x

    def parameter_rrefs(self) -> list:
        """
        Retrieve RRefs for all parameters of the model.

        Returns:
            list: A list of RRefs pointing to each parameter of the model.
        """
        remote_params = []
        remote_params.extend(_remote_method(_parameter_rrefs, self.layer_0_rref))
        remote_params.extend(_parameter_rrefs(self.layer_1))
        remote_params.extend(_remote_method(_parameter_rrefs, self.layer_2_rref))
        remote_params.extend(_parameter_rrefs(self.layer_3))
        return remote_params
