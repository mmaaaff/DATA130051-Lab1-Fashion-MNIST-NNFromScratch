import os
import cupy as np
import MyDL
import MyDL.nn as nn

from MyDL import utils

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

    def save(self, filename, path=None):
        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, filename)
        else:
            path = filename
        self.eval()
        np.savez(path, *[utils.np_get(param.data) for param in self.params])

    def load(self, path):
        with np.load(path) as weights:
            for param, (name, data) in zip(self.params, weights.items()):
                if USE_CUPY:
                    param.data = np.asarray(data)
                else:
                    param.data = data

    def __repr__(self):
        return f"nn.NeuralNetwork '{self.__class__.__name__}'"
    
    def __call__(self, x):
        return self.forward(x)
    
class Sequential():
    def __init__(self, *layers):
        self.params = []
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
    
    def __call__(self, x):
        return self.forward(x)