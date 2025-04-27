import os
import cupy as np
import MyDL
import MyDL.nn as nn
from MyDL.optimizer import Optimizer

from MyDL import utils
from typing import Tuple

USE_CUPY = True

class NeuralNetwork:
    def __init__(self):
        self.params = []
        self.train()

    def __setattr__(self, name, value):
        """
        Automatically add parameters of layers to the network param list
        """
        super().__setattr__(name, value)  # Default
        if hasattr(value, 'params'):
            if not hasattr(self, 'params'):
                self.params = []
            self.params.extend(value.params)

    def forward(self, x):
        return x

    def train(self):
        for param in self.params:
            param.requires_grad = True
        attributes = dir(self)  # return all attribute names of self
        for attr in attributes:
            if not attr.startswith('__') and not attr.endswith('__'):
                obj = getattr(self, attr)
                if hasattr(obj, 'train') and callable(getattr(obj, 'train')):
                    obj.train()

    def eval(self):
        for param in self.params:
            param.requires_grad = False
        attributes = dir(self)
        for attr in attributes:
            if not attr.startswith('__') and not attr.endswith('__'):
                obj = getattr(self, attr)
                if hasattr(obj, 'eval') and callable(getattr(obj, 'eval')):
                    obj.eval()

    def save(self, filename, path=None, optim:Optimizer=None):
        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, filename)
        else:
            path = filename
        self.eval()
        if optim:
            np.savez(path, *[utils.np_get(param.data) for param in self.params], *[utils.np_get(param.data) for param in optim.optimizer_params])
        else:
            np.savez(path, *[utils.np_get(param.data) for param in self.params])

    def load(self, path, optim:Optimizer=None):
        with np.load(path) as weights:
            weights_data = weights.values() if np.__name__ == 'numpy' else weights.npz_file.values()
            weights_data = list(weights_data)
            for param, data in zip(self.params, weights_data[:len(self.params)]):
                if param.shape != data.shape:
                    raise ValueError(f"Shape mismatch: {param.shape} vs {data.shape}")
                if USE_CUPY:
                    param.data = np.asarray(data)
                else:
                    param.data = data   
            if optim:
                for param, data in zip(optim.optimizer_params, weights_data[len(self.params):]):
                    if param.shape != data.shape:
                        raise ValueError(f"Shape mismatch: {param.shape} vs {data.shape}")
                    if USE_CUPY:
                        param.data = np.asarray(data)
                    else:
                        param.data = data

    def __repr__(self):
        return f"nn.NeuralNetwork '{self.__class__.__name__}'"
    
    def __call__(self, x):
        return self.forward(x)
    
class Sequential(nn.Layer):
    def __init__(self, *layers: Tuple[nn.Layer]):
        super().__init__()
        self.train()
        self.layers = layers
        for layer in self.layers:
            if hasattr(layer, 'params'):
                self.params.extend(layer.params)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train(self):
        attributes = dir(self)  # return all attribute names of self
        for attr in attributes:
            if not attr.startswith('__') and not attr.endswith('__'):
                obj = getattr(self, attr)
                if hasattr(obj, 'train') and callable(getattr(obj, 'train')):
                    obj.train()

    def eval(self):
        attributes = dir(self)
        for attr in attributes:
            if not attr.startswith('__') and not attr.endswith('__'):
                obj = getattr(self, attr)
                if hasattr(obj, 'eval') and callable(getattr(obj, 'eval')):
                    obj.eval()

    def __repr__(self):
        return f"nn.Sequential '{self.__class__.__name__}'"