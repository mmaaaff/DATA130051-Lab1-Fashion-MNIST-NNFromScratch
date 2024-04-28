from .tensor import *
import numpy as np

class Adam():
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, decay_rate=0.2):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(param).astype(float) for param in params]
        self.v = [np.zeros_like(param).astype(float) for param in params]
        self.t = 0
        self.decay_rate = decay_rate
    def step(self):
        self.t += 1
        lr = self.lr / (self.t**self.decay_rate)
        grad = [param.grad for param in self.params]
        self.m = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.m, grad)]
        self.v = [self.beta2 * v + (1 - self.beta2) * g**2 for v, g in zip(self.v, grad)]
        m_hat = [m / (1 - self.beta1**self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2**self.t) for v in self.v]
        update = [-lr * m / (np.sqrt(v + self.eps)) for m, v in zip(m_hat, v_hat)]
        for param, up in zip(self.params, update):
            param.data += up
    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data)