import torch
import torch.nn as nn
import torch.distributed.rpc as rpc

from torch.distributed.rpc import RRef, rpc_async, remote, rpc_sync

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

def _parameter_rrefs(module):
        param_rrefs = []
        for param in module.parameters():
            param_rrefs.append(RRef(param))
        return param_rrefs

class Layer_0(nn.Module):
    def __init__(self):
        super(Layer_0, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layers.to("cuda")
    def forward(self, x):
        return self.layers(x)

class Layer_1(nn.Module):
    def __init__(self):
        super(Layer_1, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
    
    def forward(self, x):
        return self.layers(x)
        
class Layer_2(nn.Module):
    def __init__(self):
        super(Layer_2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.layers.to("cuda")
    
    def forward(self, x):
        return self.layers(x)
    
class Layer_3(nn.Module):
    def __init__(self):
        super(Layer_3, self).__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10))
        
    def forward(self, x):
        return self.layers(x)
    
    
class SplitModel(nn.Module):
    def __init__(self, worker):
        super(SplitModel, self).__init__()
        

        # setup Layer 0 remotely
        self.layer_0_rref = rpc.remote(worker, Layer_0, args=())
        # setup Layer 1 locally
        self.layer_1 = Layer_1()
        # setup Layer 2 remotely
        self.layer_2_rref = rpc.remote(worker, Layer_2, args=())
        # setup layer 3 locally
        self.layer_3 = Layer_3()

    def forward(self, x):
        x = _remote_method(Layer_0.forward, self.layer_0_rref, x)
        x = self.layer_1(x)
        x = _remote_method(Layer_2.forward, self.layer_2_rref, x)
        x = self.layer_3(x)
        return x
    
    
    def parameter_rrefs(self):
        remote_params = []
        # get RRefs of Layer 0
        remote_params.extend(_remote_method(_parameter_rrefs, self.layer_0_rref))
        # create RRefs for local parameters
        remote_params.extend(_parameter_rrefs(self.layer_1))
        remote_params.extend(_remote_method(_parameter_rrefs, self.layer_2_rref))
        # get RRefs of decoder
        remote_params.extend(_parameter_rrefs(self.layer_3))
        return remote_params