import os
import numpy as np
import MyDL
import MyDL.nn as nn

class NeuralNetwork:
    def __init__(self):
        self.params = []
        self.train()

    def forward(self, x):
        return x

    def train(self):
        for param in self.params:
            param.requires_grad = True
        attributes = dir(self)
        for attr in attributes:
            if not attr.startswith('__') and not attr.endswith('__'):
                value = getattr(self, attr)
                if isinstance(value, nn.BatchNorm1d):
                    value.train()

    def eval(self):
        for param in self.params:
            param.requires_grad = False
        attributes = dir(self)
        for attr in attributes:
            if not attr.startswith('__') and not attr.endswith('__'):
                if isinstance(getattr(self, attr), nn.BatchNorm1d):
                    getattr(self, attr).eval()

    def save(self, filename, path=None):
        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, filename)
        else:
            path = filename
        self.eval()
        np.savez(path, *[param.data for param in self.params])

    def load(self, path):
        with np.load(path) as weights:
            for param, (name, data) in zip(self.params, weights.items()):
                param.data = data

    def __repr__(self):
        return f"NeuralNetwork '{self.__class__.__name__}'"
    
    def __call__(self, x):
        return self.forward(x)