from ..tensor import *
from typing import Union, Tuple
from cupy.lib.stride_tricks import as_strided
import time
import abc

from MyDL import utils


class Layer(abc.ABC):
    def __init__(self):
        self.params = []
        self.training = True

    @abc.abstractmethod
    def forward(self, x):
        pass

    def train(self):
        self.training = True
        for param in self.params:
            param.requires_grad = True

    def eval(self):
        self.training = False
        for param in self.params:
            param.requires_grad = False

    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Linear(Layer):
    def __init__(self, in_features, out_features, initialize='xavier'):
        super().__init__()
        if initialize == 'xavier':
            self.weights = MyTensor(np.sqrt(6 / (in_features * out_features)) * 2 * (np.random.rand(in_features, out_features) - 1))
            print(self.weights.shape)
            self.bias = MyTensor(np.zeros(out_features), requires_grad=True)
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


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.params = []
    
    @staticmethod
    def forward(x):
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
    

class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.params = []

    @staticmethod
    def forward(x):
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
    

class Conv2D(Layer):
    """
    2D convolution
    """
    def __init__(self, in_channels:   int,
                       out_channels:  int,
                       kernel_size:   Union[int, Tuple[int, int]],
                       stride:        Union[int, Tuple[int, int]],
                       padding:       Union[int, Tuple[int, int, int, int]],  # size of 0-padding
                       bias:          bool
                ):
        super().__init__()
        self.c_in = in_channels
        self.c_out = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride , stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding, padding)
        
        self.training = True

        # Using Kaiming initialization
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.kernel = MyTensor(np.sqrt(1 / fan_in) * 2 * (np.random.rand(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]) - 0.5))
        if bias:
            self.biased = True
            self.bias = MyTensor(np.sqrt(1 / fan_in) * 2 * (np.random.rand(out_channels) - 0.5))
        else:
            self.biased = False

        self.params = [self.kernel]
        if self.biased:
            self.params.append(self.bias)

    def forward(self, x:MyTensor):
        # convert x to shape b * c_in * h * w
        single_img = False
        if len(x.shape) == 3:
            single_img = True
            if x.shape[0] == self.c_in:
                x = x.up_dim(axis=0)
            else:
                raise ValueError(f"Input channel of conv2D is incorrect! Input channle should be {self.c_in} but got input x in shape of {x.shape}")
        elif len(x.shape) != 4:
            raise ValueError(f"Input of conv2D is illegal. Require bchw or chw, but got input x in shape {x.shape}")
        
        # Calculate the value of output
        x_out_data, x_pad_strided = self.conv2d_multi_channel(x.data, self.kernel.data, padding=self.padding, stride=self.stride)
        if single_img:
            x_out_data = x_out_data[0]

        # Create output tensor and properly put it into the computational graph and deliver the gradient
        x_out = MyTensor(x_out_data, requires_grad=self.training)
        x_out._cached_x_strided = x_pad_strided  # Save the strided x for gradient calculation, x_strided: (b, c_in, H_out, W_out, kH, kW)
        if self.training:  # requires gradient
            x_out.children = [x, self.kernel]
            def conv_grad_fn_backward(x_out:MyTensor):  # gradient delivering of conv operation
                """
                We calculate using the following rule:
                Let B = pad(A) stride_conv K, then:
                grad(A) = inverse_pad(stride(grad(B))) conv rot180(K_io_channel_inverse);
                grad(K) = pad(A) conv stride(grad(B))
                It's hard to handle the channel and batch when computing grad(K)
                """
                # Calculate the size if stride=1
                # start = time.perf_counter()
                C1, C2 = x_out.children[0], x_out.children[1]
                b, c_in, H, W = C1.shape
                c_out, _, kH, kW = C2.shape  
                H_out_no_stride = H + self.padding[0] + self.padding[1] - kH + 1
                W_out_no_stride = W + self.padding[2] + self.padding[3] - kW + 1

                # Compute stride(grad(B))
                x_out_stride = np.zeros((b, c_out, H_out_no_stride, W_out_no_stride))
                x_out_stride[:, :, ::self.stride[0], ::self.stride[1]] = x_out.grad
                # Compute inverse_pad(stride(grad(B)))
                x_out_stride_invpad = np.pad(x_out_stride, ((0, 0), 
                                                            (0, 0), 
                                                            (self.kernel_size[0] - 1 - self.padding[0], 
                                                             self.kernel_size[0] - 1 - self.padding[1]),
                                                            (self.kernel_size[1] - 1 - self.padding[2], 
                                                             self.kernel_size[1] - 1 - self.padding[3])),
                                                            mode='constant')
                # Compute rot180(K_io_channel_inverse)
                K_rot180_ioc_inverse = np.rot90(np.swapaxes(C2.data, 0, 1), k=2, axes=(2,3))
                # Compute gradient
                # end = time.perf_counter()
                # print(f"Time 1: {end - start:.4f} seconds")
                # start = time.perf_counter()
                if C1.requires_grad:
                    x_grad_local, _ = self.conv2d_multi_channel(x_out_stride_invpad, K_rot180_ioc_inverse)
                    C1.grad += x_grad_local
                # end = time.perf_counter()
                # print(f"Time C1: {end - start:.4f} seconds")
                # start = time.perf_counter()
                if C2.requires_grad:
                    # kernel_grad_local, _ = self.conv2d_multi_channel(np.expand_dims(x_pad, axis=2), np.expand_dims(x_out_stride, axis=2), extra_dim=True)
                    # kernel_grad_local = np.sum(np.swapaxes(kernel_grad_local, 1, 2), axis=0)
                    kernel_grad_local = np.einsum('bihwmn,bohw->oimn', x_out._cached_x_strided, x_out.grad)
                    C2.grad += kernel_grad_local
                # end = time.perf_counter()
                # print(f"Time C2: {end - start:.4f} seconds")

            x_out.add_grad_fn(conv_grad_fn_backward)
        
        if self.biased:
            x_out = x_out + self.bias.up_dim(axis=(0,2,3))
        return x_out

    @staticmethod
    def conv2d_multi_channel(X, K, padding:Union[int, Tuple[int, int, int, int]]=0, 
                         stride:Union[int, Tuple[int, int]]=1, extra_dim=False):
        """
        2D-conv.
        Args:
            X: Input image (batch, C_in, H, W)
            K: Convolutional kernel (C_out, C_in, kH, kW)
            padding: Size of padding (up, down, left, right)
            stride: Marching stride of kernel while convoluting
            extra_dim: For special cases, X is in size of (B, batch, C_in, H, W), 
                       and kernel is (B, C_out, C_in, kH, kW)). 
                       This happens when calculating gradient, and needs proper handling
        Returns:
            Y: Output featuremap (C_out, H_out, W_out)
        """
        if extra_dim == False:
            batch, C_in, H, W = X.shape
            C_out, C_in_k, kH, kW = K.shape
            assert C_in == C_in_k, "Input channel of x equals that of the kernel"

            # padding each dim
            if isinstance(padding, int):
                X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
            else:
                X_padded = np.pad(X, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[2], padding[3])), mode='constant')
            
            if isinstance(stride, int):
                stride = (stride, stride)

            H_p, W_p = X_padded.shape[2], X_padded.shape[3]

            H_out = (H_p - kH) // stride[0] + 1
            W_out = (W_p - kW) // stride[1] + 1

            # Unfolded X. The original single element is now a convolutional area
            strides = X_padded.strides
            new_shape = (batch, C_in, H_out, W_out, kH, kW)
            new_strides = (
                strides[0],
                strides[1],
                strides[2] * stride[0],
                strides[3] * stride[1],
                strides[2],
                strides[3]
            )
            X_strided = as_strided(X_padded, shape=new_shape, strides=new_strides)

            # K: (C_out, C_in, kH, kW), X: (batch, C_in, H_out, W_out, kH, kW)
            # use einsum for efficient sum
            Y = np.einsum('oimn,bihwmn->bohw', K, X_strided)
        
        else:
            B, batch, C_in, H, W = X.shape
            B, C_out, C_in_k, kH, kW = K.shape
            assert C_in == C_in_k, "Input channel of x equals that of the kernel"

            # padding each dim
            if isinstance(padding, int):
                X_padded = np.pad(X, ((0, 0), (0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
            else:
                X_padded = np.pad(X, ((0, 0), (0, 0), (0, 0), (padding[0], padding[1]), (padding[2], padding[3])), mode='constant')
            
            if isinstance(stride, int):
                stride = (stride, stride)

            H_p, W_p = X_padded.shape[3], X_padded.shape[4]

            H_out = (H_p - kH) // stride[0] + 1
            W_out = (W_p - kW) // stride[1] + 1

            # Unfolded X. The original single element is now a convolutional area
            strides = X_padded.strides
            new_shape = (B, batch, C_in, H_out, W_out, kH, kW)
            new_strides = (
                strides[0],  # B
                strides[1],  # batch
                strides[2],  # C
                strides[3] * stride[0],  # H
                strides[4] * stride[1],  # W
                strides[3],  # H
                strides[4]  # W
            )
            X_strided = as_strided(X_padded, shape=new_shape, strides=new_strides)

            # K: (B, C_out, C_in, kH, kW), X: (B, batch, C_in, H_out, W_out, kH, kW)
            # use einsum for efficient sum
            Y = np.einsum('Boimn,Bbihwmn->Bbohw', K, X_strided)

        return Y, X_strided
        

class FullAveragePool2d(Layer):
    def __init__(self):
        super().__init__()
        self.params = []

    @staticmethod
    def forward(x:MyTensor):
        assert len(x.shape) == 4, f"Require x in dim 4, but got x.shape = {x.shape}"
        x = x.sum(axis=(2, 3)).lower_dim(axis=(2, 3)) * (1 / (x.shape[2] * x.shape[3]))
        return x


class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.params = []

    @staticmethod
    def forward(x, dim=-1):
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


class BatchNorm1d(Layer):
    def __init__(self, length):
        super().__init__()
        self.u_all = []  # Record the mean of each batch
        self.var_all = []  # Record the variance of each batch
        self.mean = MyTensor(np.zeros(length), requires_grad=False)
        self.variance = MyTensor(np.zeros(length), requires_grad=False)
        self.num_sample_passed = 0
        self.training = True
        self.eps = 1e-8
        self.params = [self.mean, self.variance]
    def forward(self, x):
        '''
        In this section, we distinguish between training and testing.
        While training, we use batch normalization and record the mean and variance of each batch.
        While testing, we apply total variance formula to calculate overall variance. Mean is calculated by taking mean of all record of u_all.
        '''
        if x.shape[0] == None:
            raise ValueError("Cannot perform BatchNorm transformation on scalar tensor")
        elif len(x.shape) == 1:
            x = x.up_dim()
        if self.training:  # During training, use batch normalization and record the mean and variance of each batch
            # x: (b, c)
            u = x.sum(axis=0) * (1 / x.shape[0])  # (1, c)
            u = u[0]  # squeeze dim, (c)
            var = ((x - u).square().sum(axis=0) * (1 / x.shape[0]))[0]  # (c)
            x = (x - u) * ((var + self.eps).sqrt().inv())
            u_data = u.data
            self.u_all.append(u_data)
            self.var_all.append(var.data)
            self.num_sample_passed += x.shape[1]
        else:
            x = (x - self.mean) * ((self.variance + self.eps).sqrt().inv())
        return x
    def eval(self):
        self.training = False
        u_all = np.array(self.u_all)
        var_all = np.array(self.var_all)
        u_variance = u_all.var(axis=0)  # (c)
        var_mean = var_all.mean(axis=0)  # (c)
        var_total = var_mean + u_variance
        u_total = u_all.mean(axis=0)
        self.variance.data = var_total
        self.mean.data = u_total
        pass
    def train(self):
        self.training = True
        self.variance.requires_grad = False
        self.mean.requires_grad = False
    
class BatchNorm2d(Layer):
    def __init__(self, channel):
        super().__init__()
        self.u_all = []  # Record the mean of each batch
        self.var_all = []  # Record the variance of each batch
        self.mean = MyTensor(np.zeros((1, channel, 1, 1)), requires_grad=False)
        self.variance = MyTensor(np.zeros((1, channel, 1, 1)), requires_grad=False)
        self.num_sample_passed = 0  # ?
        self.training = True
        self.eps = 1e-8
        self.params = [self.mean, self.variance]
    def forward(self, x):
        assert len(x.shape) == 4, f"BatchNorm2d() requires x in dim of 4, but got x.shape = {x.shape}"
        if self.training:  # During training, use batch normalization and record the mean and variance of each batch
            u = x.sum(axis=(0, 2, 3)) * (1 / (x.shape[0] * x.shape[2] * x.shape[3]))  # (1, c, 1, 1)
            u = u.detatch()
            var = ((x - u).square().sum(axis=(0, 2, 3)) * (1 / (x.shape[0] * x.shape[2] * x.shape[3])))  # (1, c, 1, 1)
            var = var.detatch()
            x = (x - u) * ((var + self.eps).sqrt().inv())
            u_data = u.data
            self.u_all.append(u_data)
            self.var_all.append(var.data)
            self.num_sample_passed += x.shape[1]
        else:
            x = (x - self.mean) * ((self.variance + self.eps).sqrt().inv())
        return x
    def eval(self):
        self.training = False
        u_all = np.array(self.u_all)  # (n, 1, c, 1, 1)
        var_all = np.array(self.var_all)  # (n, 1, c, 1, 1)
        u_variance = u_all.var(axis=0)  # (1, c, 1, 1)
        var_mean = var_all.mean(axis=0)  # (1, c, 1, 1)
        var_total = var_mean + u_variance  # (1, c, 1, 1)
        u_total = u_all.mean(axis=0)  # (1, c, 1, 1)
        self.variance.data = var_total
        self.mean.data = u_total
        self.variance.requires_grad = False
        self.mean.requires_grad = False
    def train(self):
        self.training = True

class Dropout(Layer):
    def __init__(self, p:float=0.5):
        super().__init__()
        self.p = p

    def forward(self, x:MyTensor):
        assert len(x.shape) == 2, f"Dropout() requires x in dim of 2 (batch, vec_length), but got x.shape = {x.shape}"
        if self.training:
            mask = np.random.rand(*x.shape) < self.p
            mask = MyTensor(mask, requires_grad=False)
            x = x * mask * (1 / self.p)
        return x