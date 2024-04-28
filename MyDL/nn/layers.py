from ..tensor import *

class Linear():
    def __init__(self, in_features, out_features, initialize='random'):
        if initialize == 'random':
            self.weights = MyTensor(np.random.randn(in_features, out_features), requires_grad=True)
            self.bias = MyTensor(np.random.randn(out_features), requires_grad=True)
        elif initialize == 'zeros':
            self.weights = MyTensor(np.zeros((in_features, out_features)), requires_grad=True)
            self.bias = MyTensor(np.zeros(out_features), requires_grad=True)
        self.params = [self.weights, self.bias]  # Record the parameters of the layer, \
        # which will be added to the model's parameter list after the layer is added to the model
    def forward(self, x):
        if x.shape[0] == None:
            raise ValueError("Cannot perform linear transformation on scalar tensor")
        elif len(x.shape) == 1:
            x = x.up_dim()
        return matmul(x, self.weights) + self.bias
    def __call__(self, x):
        return self.forward(x)


class ReLU():
    def __init__(self):
        self.params = []
    def forward(self, x):
        new_x_data = abs(x.data * (x.data > 0))
        requires_grad = x.requires_grad
        x_new = MyTensor(new_x_data, requires_grad=requires_grad)
        if requires_grad:
            x_new.children = [x]
            def relu_grad_fn_backward(self):
                grad = self.grad
                local = self.children[0].data > 0
                self.children[0].grad += grad * local
            x_new.add_grad_fn(relu_grad_fn_backward)
        return x_new
    def __call__(self, x):
        return self.forward(x)
    

class Tanh():
    def __init__(self):
        self.params = []
    def forward(self, x):
        new_x_data = np.tanh(x.data)
        requires_grad = x.requires_grad
        x_new = MyTensor(new_x_data, requires_grad=requires_grad)
        if requires_grad:
            x_new.children = [x]
            def tanh_grad_fn_backward(self):
                grad = self.grad
                local = 1 - self.data**2
                self.children[0].grad += grad * local
            x_new.add_grad_fn(tanh_grad_fn_backward)
        return x_new
    def __call__(self, x):
        return self.forward(x)

        


class Softmax():
    def __init__(self):
        self.params = []
    def forward(self, x, dim=-1):
        if not isinstance(x, MyTensor):
            raise TypeError("Input for softmax layer must be a tensor")
        if x.shape == (None, ):
            raise ValueError("Cannot perform softmax on scalar tensor")
        if len(x.shape) < dim + 1:
            raise ValueError("Invalid dimension for softmax")
        if dim == -1:
            dim = len(x.shape) - 1
        max_x = MyTensor(np.max(x.data, axis=dim, keepdims=True), requires_grad=False)
        x = x - max_x  # Avoid overflow
        exp_x = exp(x)
        sum_exp_x = exp_x.sum(axis=dim)
        if sum_exp_x.shape[0] == 1:  # This means the input has only one sample(note that sum() in MyTensor will not reduce dimension)
            sum_exp_x = sum_exp_x[0]
        result = exp_x * sum_exp_x.inv()
        return result
    def __call__(self, x, dim=-1):
        return self.forward(x, dim)


class BatchNorm1d():
    def __init__(self):
        self.params = []
    def forward(self, x):
        eps = 1e-5
        if x.shape[0] == None:
            raise ValueError("Cannot perform BatchNorm transformation on scalar tensor")
        elif len(x.shape) == 1:
            x = x.up_dim()
        u = x.sum(axis=0) * (1 / x.shape[0])
        u = u[0]
        var = ((x - u).square().sum(axis=0) * (1 / x.shape[0]))[0]
        x = (x - u) * ((var + eps).sqrt().inv())
        return x
    def __call__(self, x):
        return self.forward(x)


class Dropout():
    pass


