import os
import numpy as np
import MyDL

class NeuralNetwork:
    def __init__(self):
        self.params = []
        self.train()

    def forward(self, x):
        return x

    def train(self):
        for param in self.params:
            param.requires_grad = True

    def eval(self):
        for param in self.params:
            param.requires_grad = False

    def save(self, filename, path=None):
        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, filename)
        else:
            path = filename
        param_list = [param.data for param in self.params]
        np.savez(path, *[param.data for param in self.params])

    def load(self, path):
        with np.load(path) as weights:
            for param, (name, data) in zip(self.params, weights.items()):
                param.data = data

    def __repr__(self):
        return f"NeuralNetwork '{self.__class__.__name__}'"
    
    def __call__(self, x):
        return self.forward(x)