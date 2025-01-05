import torch
import torch.nn.functional as F
from torch import nn


class Linear:

    def __init__(self, size: tuple, weight=None, bias=None, device='cpu', init_type='uniform', activation_param=0.001,
                 normalize_forward=True):

        self.device = device
        # FIXME
        # actually incorrect init
        if init_type == 'leaky_relu':

            if normalize_forward:
                stdv = torch.nn.init.calculate_gain(init_type, activation_param) * size[1] ** -0.5
            else:
                stdv = torch.nn.init.calculate_gain(init_type, activation_param) * size[0] ** -0.5

            self.weight = torch.empty(size).normal_(0, stdv).to(device)
            self.bias = torch.empty(size[1]).normal_(0, stdv).to(device)

        elif init_type == 'uniform':
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            torch.manual_seed(0)
            self.weight = torch.empty(size[0], size[1]).to(device)
            torch.nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

            self.bias = torch.empty(size[1]).to(device)
            bound = 1 / size[0] ** 0.5 if size[0] > 0 else 0
            torch.manual_seed(0)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        if weight is not None:

            self.weight = weight.detach()
            self.bias = bias.detach()

            if weight.shape != size:
                raise Exception("Loaded weight has wrong size!")

        self.out = None
        self.inp = None

    def update_weights(self, dweights, dbias):

        with torch.no_grad():
            self.weight -= dweights
            self.bias -= dbias

    def get_weights(self):
        return self.weight, self.bias

    def __call__(self, x, *args, **kwargs):

        self.inp = x

        with torch.no_grad():
            out = x @ self.weight + self.bias
            self.out = out

        return out

    def backward(self, dloss, lr, use_old, reg_lambda, regularizer):

        with torch.no_grad():
            d_weights = self.inp.T @ dloss
            d_bias = (torch.ones(self.inp.shape[0]).to(self.device) @ dloss) / self.inp.shape[0]

        old_w = self.weight.clone()

        self.update_weights(lr * (d_weights + reg_lambda * (regularizer(self.weight))),
                            lr * (d_bias + reg_lambda * (regularizer(self.bias))))

        if use_old:
            dloss @= old_w.T

        else:
            dloss @= self.weight.T

        del d_bias
        del old_w

        return dloss


class BatchNorm1d:

    def __init__(self, dim, device='cpu', eps=1e-5, momentum=0.1):

        self.eps = eps
        self.momentum = momentum

        self.weight = torch.ones(dim).to(device)
        self.bias = torch.zeros(dim).to(device)

        self.running_mean = torch.zeros(dim).to(device)
        self.running_var = torch.ones(dim).to(device)

        self.x = None
        self.xhat = None
        self.out = None

        self.mean = None
        self.var = None

        self.device = device

    def update_weights(self, dweight, dbias):
        with torch.no_grad():
            self.weight -= dweight
            self.bias -= dbias

    def __call__(self, x, is_training=True):

        self.x = x
        with torch.no_grad():
            if is_training:
                self.mean = x.mean(0, keepdim=True)  # batch mean
                self.var = x.var(0, keepdim=True, unbiased=False)  # batch variance

            else:
                self.mean = self.running_mean
                self.var = self.running_var

            self.xhat = (x - self.mean) / torch.sqrt(self.var + self.eps)  # normalize to unit variance

        if is_training:
            with torch.no_grad():
                # refer to https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d
                xvar2ema = x.var(0, keepdim=True, unbiased=True)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar2ema

        self.out = self.weight * self.xhat + self.bias

        return self.out

    def backward(self, dloss, lr, use_old, reg_lambda, regularizer):

        dL_dweight = torch.sum(dloss * self.xhat, axis=0)
        dL_dbias = torch.sum(dloss, axis=0)

        N = self.xhat.shape[0]

        if not use_old:
            self.update_weights(lr * (dL_dweight + reg_lambda * regularizer(self.weight)),
                                lr * (dL_dbias + reg_lambda * regularizer(self.bias)))

        dL_dx_hat = dloss * self.weight
        dL_dvariance = torch.sum(dL_dx_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps) ** -1.5, axis=0)
        dL_dmean = torch.sum(dL_dx_hat * -1 / torch.sqrt(self.var + self.eps), axis=0) + dL_dvariance * torch.mean(
            -2 * (self.x - self.mean), axis=0)

        dloss = (dL_dx_hat / torch.sqrt(self.var + self.eps)) + (dL_dvariance * 2 / N * (self.x - self.mean)) + (
                    dL_dmean / N)

        if use_old:
            self.update_weights(lr * (dL_dweight + reg_lambda * regularizer(self.weight)),
                                lr * (dL_dbias + reg_lambda * regularizer(self.bias)))

        return dloss


class LReLU:
    def __init__(self, a):
        self.a = a

        self.out = None
        self.local_grad = None

    def __call__(self, x, is_training=True, *args, **kwargs):
        out = torch.where(x >= 0, x, self.a * x)
        self.out = out
        self.local_grad = torch.where(out >= 0, 1, self.a)

        return out

    def backward(self, dloss, *args, **kwargs):
        dloss = dloss * self.local_grad

        del self.local_grad
        del self.out

        return dloss


class CrossEntropy:

    def __init__(self, optimize_exponents=True):
        self.optimize_exponents = optimize_exponents
        self.local_grad = None

    def __call__(self, x, target, is_training=True, *args, **kwargs):
        num_classes = x.shape[-1]
        one_hot_target = torch.nn.functional.one_hot(target.long(), num_classes)

        # if is_training:
        self.local_grad = -(one_hot_target - self.get_probabilities(x)) / len(x)
        mean_cross_entropy = -(
                    (x * one_hot_target).sum(axis=1, keepdim=True) - x.exp().sum(axis=1, keepdim=True).log()).mean()
        return mean_cross_entropy

        # equal to torch.nn.CrossEntropyLoss()
        # torch_ce = torch.nn.CrossEntropyLoss()
        # return torch_ce(x, target.long())
        # assert torch.abs(torch_ce(x, target) - mean_cross_entropy).max() < 1e-3

    def get_probabilities(self, x):
        _gamma = torch.amax(x)
        if self.optimize_exponents:
            _exp_outputs = (x - _gamma).exp()
        else:
            _exp_outputs = x.exp()
        return (_exp_outputs.T / _exp_outputs.sum(axis=1)).T


class Conv2DWrapper:
    def __init__(self, in_channels, out_channels, kernel_size, device='cpu', stride=1, padding=0, bias_use=True,
                 weight=None, bias=None):
        # Initialize the Conv2D layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias_use).to(device)

        if weight is not None:
            self.conv.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.conv.bias = torch.nn.Parameter(bias, requires_grad=False)

        self.inp = None  # To store the input during forward pass
        self.out = None  # To store the output during forward pass

    def __call__(self, x, *args, **kwargs):
        # Forward pass
        self.inp = x.clone().detach()

        with torch.no_grad():
            self.out = self.conv(x)

        return self.out

    def update_weights(self, dweights, dbias, lr):
        with torch.no_grad():
            self.conv.weight -= lr * dweights
            if self.conv.bias is not None:
                self.conv.bias -= lr * dbias

    def backward(self, dloss, lr, use_old, *args, **kwargs):

        dweights = F.conv2d(self.inp.permute(1, 0, 2, 3), dloss.permute(1, 0, 2, 3), stride=self.conv.stride,
                            padding=self.conv.padding)
        dweights = dweights.permute(1, 0, 2, 3)  # Permute back to match weight shape

        if self.conv.bias is not None:
            dbias = dloss.sum(dim=(0, 2, 3))  # Sum over batch, height, and width
        else:
            dbias = None

        if use_old:
            dloss_prev = F.conv_transpose2d(dloss, self.conv.weight, stride=self.conv.stride, padding=self.conv.padding)
            self.update_weights(dweights, dbias, lr)
            del dweights
            del dbias
            return dloss_prev
        else:
            self.update_weights(dweights, dbias, lr)
            dloss_prev = F.conv_transpose2d(dloss, self.conv.weight, stride=self.conv.stride, padding=self.conv.padding)
            del dweights
            del dbias
            return dloss_prev

        # print(f'{type(self)=}')
        # print(f'{dweights.shape=}')
        # print(f'{self.conv.weight.shape=}')


class MyFlatten:
    def __init__(self):
        self.input_shape = None

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        self.input_shape = x.shape
        return torch.flatten(x, 1)

    def backward(self, dloss: torch.Tensor, *args, **kwargs):
        return dloss.reshape(self.input_shape)


class MaxPool2DWrapper:
    def __init__(self, kernel_size, stride, padding):
        self.inp = None  # To store the input during forward pass
        self.out = None  # To store the output during forward pass
        self.indices = None  # To store the indices of max values during forward pass

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x, *args, **kwargs):
        # Forward pass
        self.inp = x.clone()
        with torch.no_grad():
            self.out, self.indices = F.max_pool2d(x, self.kernel_size, self.stride, self.padding, return_indices=True)
        return self.out

    def backward(self, dloss, *args, **kwargs):
        dloss_prev = F.max_unpool2d(dloss, self.indices, self.kernel_size, self.stride, self.padding, self.inp.shape)
        del self.inp
        return dloss_prev