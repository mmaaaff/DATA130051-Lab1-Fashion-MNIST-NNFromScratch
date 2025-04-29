from .tensor import *
import cupy as np
import abc


class Optimizer(abc.ABC):
    def __init__(self, params, lr):
        self.params = params
        self.optimizer_params = []
        self.t = 0
        self.lr = lr

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def zero_grad(self):
        pass

    def empty(self):
        for param in self.params:
            param.data = None
            param.grad = None

class Scheduler(abc.ABC):
    @abc.abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, lr)

        self.optimizer_params = [self.lr]
        
    def step(self):
        for param in self.params:
            if param.requires_grad:
                param.data -= self.lr * param.grad
        self.t += 1

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(np.array(param.data))


class Momentum(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum
        self.v = [np.zeros_like(np.array(param.data)).astype(float) for param in params]

        self.optimizer_params = [self.lr, self.momentum, self.v]

    def step(self):
        for i, param in enumerate(self.params):
            if param.requires_grad:
                self.v[i] = self.momentum * self.v[i] - self.lr * param.grad
                param.data += self.v[i]
        self.t += 1

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(np.array(param.data))


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, decay_rate=0.2):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(np.array(param.data)).astype(float) for param in params]
        self.v = [np.zeros_like(np.array(param.data)).astype(float) for param in params]
        self.decay_rate = decay_rate

        self.optimizer_params = [self.lr, self.beta1, self.beta2, self.eps, self.decay_rate, self.m, self.v]

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
            if param.requires_grad:
                param.data += up

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(np.array(param.data))


class MultiStepLR(Scheduler):
    def __init__(self, optimizer:Optimizer, milestones:list|tuple, gamma:int):
        assert len(milestones) > 0, "milestones should be a list not empty"
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma

    def step(self):
        if len(self.milestones) > 0:
            if self.optimizer.t == self.milestones[0]:
                self.optimizer.lr *= self.gamma
                print(f'Leaening rate set to {self.optimizer.lr}')
                self.milestones.pop(0)