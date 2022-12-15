"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            in_features, out_features, device=device, dtype=dtype, requires_grad=True
        ))
        self.bias = Parameter(init.kaiming_uniform(
            out_features, 1, device=device, dtype=dtype, requires_grad=True
        ).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Z = X @ self.weight
        if self.bias is not None:
            Z += self.bias.broadcast_to(Z.shape)
        return Z
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch, dim = X.shape[0], np.prod(X.shape[1:])
        return X.reshape((batch, dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n = logits.shape[-1]
        batch = logits.shape[0]
        axes = tuple(list(range(1, len(logits.shape))))
        loss = ops.logsumexp(logits, axes=axes) - \
            ops.summation(logits * init.one_hot(n, y,
                                                device=logits.device, dtype=logits.dtype), axes=axes)
        return loss.sum() / batch
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(
            dim, device=device, dtype=dtype, requires_grad=True))
        # running mean and running var not parameters
        self.running_mean = Tensor(init.zeros(dim, device=device, dtype=dtype))
        self.running_var = Tensor(init.ones(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, dim = x.shape
        if self.training:
            mean = ops.summation(x, axes=(0,)) / batch  # (dim, )
            mean_bc = mean.reshape((1, dim)).broadcast_to(
                x.shape)  # (batch, dim)
            var = ops.summation((x - mean_bc) ** 2,
                                axes=(0,)) / batch  # (dim, )
            var_bc = var.reshape((1, dim)).broadcast_to(
                x.shape)  # (batch, dim)

            self.running_mean = (1 - self.momentum) * \
                self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * \
                self.running_var + self.momentum * var.detach()
        else:
            mean_bc = self.running_mean.detach().reshape((1, dim)).broadcast_to(x.shape)
            var_bc = self.running_var.detach().reshape((1, dim)).broadcast_to(x.shape)

        x_norm = (x - mean_bc) / ((var_bc + self.eps) ** 0.5)  # (batch, dim)
        y = self.weight.reshape((1, dim)).broadcast_to(
            x.shape) * x_norm + self.bias.reshape((1, dim)).broadcast_to(x.shape)
        return y
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose(
            (2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(
            1, dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, dim = x.shape
        mean = ops.summation(x, axes=(1,)) / self.dim  # (batch, )
        mean = mean.reshape((batch, 1)).broadcast_to(x.shape)  # (batch, dim)
        var = ops.summation((x - mean) ** 2, axes=(1,)) / self.dim  # (batch, )
        var = var.reshape((batch, 1)).broadcast_to(x.shape)  # (batch, dim)
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)  # (batch, dim)
        y = self.weight.broadcast_to(
            x.shape) * x_norm + self.bias.broadcast_to(x.shape)
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p,
                              device=x.device, dtype=x.dtype)
            x = x * mask / (1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            (kernel_size ** 2) * in_channels,
            (kernel_size ** 2) * out_channels,
            shape=(kernel_size, kernel_size, in_channels, out_channels),
            device=device, dtype=dtype, requires_grad=True
        ))
        bound = 1 / math.sqrt((kernel_size ** 2) * in_channels)
        self.bias = Parameter(init.rand(
            out_channels, low=-bound, high=bound,
            device=device, dtype=dtype, requires_grad=True
        )) if bias else None

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # N, C, H, W = x.shape

        # N, H, W, C
        x = ops.transpose(ops.transpose(x, axes=(1, 2)), axes=(2, 3))

        # N, H_o, W_o, C_o
        out = ops.conv(x, self.weight, stride=self.stride,
                       padding=(self.kernel_size)//2)

        if self.bias:
            out += self.bias.reshape(
                (1, 1, 1, self.out_channels)
            ).broadcast_to(out.shape)

        # # N, C_o, H_o, W_o
        out = ops.transpose(ops.transpose(out, axes=(2, 3)), axes=(1, 2))

        # Question: if transpose before adding bias, Resnet9 is wrong, why
        # if self.bias:
        #     out += self.bias.reshape(
        #         (1, self.out_channels, 1, 1)
        #         ).broadcast_to(out.shape)

        return out
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.activation = Tanh() if nonlinearity == 'tanh' else ReLU()
        self.bias = bias
        k = 1. / math.sqrt(hidden_size)
        self.W_ih = Parameter(init.rand(
            input_size, hidden_size, low=-k, high=k,
            device=device, dtype=dtype, requires_grad=True
        ))
        self.W_hh = Parameter(init.rand(
            hidden_size, hidden_size, low=-k, high=k,
            device=device, dtype=dtype, requires_grad=True
        ))
        self.bias_ih = Parameter(init.rand(
            hidden_size, low=-k, high=k,
            device=device, dtype=dtype, requires_grad=True
        )) if bias else None
        self.bias_hh = Parameter(init.rand(
            hidden_size, low=-k, high=k,
            device=device, dtype=dtype, requires_grad=True
        )) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ## BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size,
                           dtype=X.dtype, device=X.device)

        Z = X @ self.W_ih + h @ self.W_hh
        # bs, hidden_size

        if self.bias_ih:
            Z += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size)) + \
                self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(
                    (bs, self.hidden_size))

        h_n = self.activation(Z)
        return h_n
        ## END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_cells = [
            RNNCell(input_size, hidden_size, bias=bias,
                    nonlinearity=nonlinearity, device=device, dtype=dtype)
        ]
        for _ in range(num_layers - 1):
            self.rnn_cells.append(
                RNNCell(hidden_size, hidden_size, bias=bias,
                        nonlinearity=nonlinearity, device=device, dtype=dtype)
            )
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        X_list = ops.split(X, axis=0)

        out_list = []
        if h0:
            h_list = list(ops.split(h0, axis=0))
        else:
            h_list = [None] * self.num_layers

        for t in range(seq_len):
            X_in = X_list[t]
            for i, rnn_cell in enumerate(self.rnn_cells):
                h = rnn_cell(X_in, h_list[i])
                X_in = h
                h_list[i] = h
            out_list.append(X_in)

        return ops.stack(out_list, axis=0), ops.stack(h_list, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
