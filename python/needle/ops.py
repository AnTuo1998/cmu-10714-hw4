"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # return (out_grad * self.scalar * (a ** self.scalar - 1))
        return out_grad * (a ** (self.scalar - 1) * self.scalar)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs,  -out_grad * lhs * (rhs ** (-2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        num_dim = len(a.shape)
        axes = list(range(num_dim))
        dim1, dim2 = self.axes if self.axes is not None else (
            num_dim-2, num_dim-1)
        axes[dim1], axes[dim2] = axes[dim2], axes[dim1]
        # return array_api.transpose(a, axes)
        return a.permute(axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        out_shape = out_grad.shape
        num_in_dim, num_out_dim = len(in_shape), len(out_shape)
        in_shape = [1] * (num_out_dim - num_in_dim) + list(in_shape)
        sum_axes = []
        for idx, (in_dim, out_dim) in enumerate(zip(in_shape, out_shape)):
            if in_dim != out_dim:
                sum_axes.append(idx)
        return reshape(summation(out_grad, axes=tuple(sum_axes)), shape=tuple(in_shape))
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # print("sum", a.shape, self.axes)
        # print("sum", array_api.sum(a, axis=self.axes).shape)
        # special case sum(axes=()) for no sum
        if isinstance(self.axes, tuple) and not self.axes:
            return a
        return a.sum(axis=self.axes)
        ### END YOUR SOLUTION
        # if self.axes is None:
        #     return a.sum()
        # else:
        #     # NOTE self.axes maybe int
        #     if isinstance(self.axes, int):
        #         return a.sum(self.axes)
        #     # NOTE only support sum in a single dim
        #     for i, axis in enumerate(sorted(list(self.axes))):
        #         # NOTE -i because each sum() operation will reduce the dimension number
        #         a = a.sum(axis-i)
        #     return a

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        num_in_dim = len(in_shape)

        if self.axes is None:
            reshape_shape = [1] * num_in_dim
        elif isinstance(self.axes, int):
            reshape_shape = list(in_shape)
            reshape_shape[self.axes] = 1
        else:
            axes = list(map(lambda x: x if x >= 0 else num_in_dim - x,
                            self.axes))
            reshape_shape = list(in_shape)
            for axis in axes:
                reshape_shape[axis] = 1
        return broadcast_to(reshape(out_grad, reshape_shape), in_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)
        if len(lhs_grad.shape) > len(lhs.shape):
            axes = tuple(range(len(lhs_grad.shape) - len(lhs.shape)))
            lhs_grad = summation(lhs_grad, axes=axes)
        if len(rhs_grad.shape) > len(rhs.shape):
            axes = tuple(range(len(rhs_grad.shape) - len(rhs.shape)))
            rhs_grad = summation(rhs_grad, axes=axes)
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -1 * out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        relu_grad = node.numpy() > 0
        return out_grad * Tensor(relu_grad, dtype="float32", device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=self.axes, keepdims=True)
        max_z_bc = array_api.broadcast_to(max_z, Z.shape)
        # max_z = array_api.squeeze(max_z, axes=self.axes)
        max_z = Z.max(axis=self.axes)
        return array_api.log(array_api.exp(Z - max_z_bc).sum(self.axes)) + max_z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        z_shape = Z.shape
        num_in_dim = len(z_shape)
        if self.axes is None:
            bc_shape = [1] * len(z_shape)
        elif isinstance(self.axes, int):
            bc_shape = list(z_shape)
            bc_shape[self.axes] = 1
        else:
            axes = list(map(lambda x: x if x >= 0 else num_in_dim - x,
                            self.axes))
            bc_shape = list(z_shape)
            for axis in axes:
                bc_shape[axis] = 1

        out_grad_bc = out_grad.reshape(bc_shape).broadcast_to(Z.shape)
        # math trick here
        return out_grad_bc * exp(Z - node.reshape(bc_shape).broadcast_to(Z.shape))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - node ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        dtype = args[0].dtype
        device = args[0].device
        shape = list(args[0].shape)
        new_shape = shape[:self.axis] + [len(args)] + shape[self.axis:]
        reshape_shape = shape[:self.axis] + [1] + shape[self.axis:]

        out = array_api.empty(
            new_shape, dtype=dtype, device=device)
        for i, arg in enumerate(args):
            arg = arg.reshape(reshape_shape)
            idx = [slice(None, None, None)] * len(new_shape)
            idx[self.axis] = i
            out[tuple(idx)] = arg.compact()

        return out.compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        assert len(A.shape) > self.axis
        shape = list(A.shape)
        new_shape = shape[:self.axis] + shape[(self.axis + 1):]
        res = []
        for i in range(0, shape[self.axis], 1):
            idx = [slice(None, None, None)] * len(shape)
            idx[self.axis] = i
            # compact is a must
            res.append(A[tuple(idx)].compact().reshape(new_shape))
            # res.append(A[tuple(idx)].sum(self.axis))

        return tuple(res)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(tuple(out_grad), self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes=None):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = a.shape
        new_shape = list(shape)
        if self.axes is None:
            axes = range(len(a.shape))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        idx = [slice(None, None, None)] * len(new_shape)
        for axis in axes:
            if axis >= len(new_shape):
                return a
            new_shape[axis] = shape[axis] * (self.dilation + 1)
            idx[axis] = slice(0, new_shape[axis], self.dilation + 1)

        out = array_api.full(
            new_shape, 0, dtype=a.dtype, device=a.device)
        out[tuple(idx)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = a.shape
        new_shape = list(shape)
        if self.axes is None:
            axes = range(len(a.shape))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        idx = [slice(None, None, None)] * len(shape)
        for axis in axes:
            if axis >= len(new_shape):
                return a
            idx[axis] = slice(0, shape[axis], self.dilation + 1)

        return a[tuple(idx)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        pad_axes = ((0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding), (0, 0))

        # need compact to make gradient work
        A = A.compact().pad(pad_axes)
        B = B.compact()
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        H_out, W_out = (H - K) // self.stride + 1, (W - K) // self.stride + 1

        # new_shape = (N, H - K + 1, W - K + 1, K, K, C_in)
        # new_strides = (Ns, Hs, Ws, Hs, Ws, Cs)
        # out_shape = (N, H - K + 1, W - K + 1, C_out)
        new_shape = (N, H_out, W_out, K, K, C_in)
        new_strides = (Ns, (Hs * self.stride), (Ws * self.stride), Hs, Ws, Cs)
        out_shape = (N, H_out, W_out, C_out)

        outer_dim = new_shape[0] * new_shape[1] * new_shape[2]
        inner_dim = K * K * C_in
        # A = NDArray.make(
        #     new_shape,
        #     strides = new_strides,
        #     device = A.device,
        #     handle = A._handle,
        #     offset = A._offset
        # ).compact()
        A = A.as_strided(new_shape, new_strides).compact()
        A = A.reshape((outer_dim, inner_dim))
        out = A @ (B.reshape((K*K*C_in, C_out)))
        return out.reshape(out_shape)
        # return out.reshape(out_shape)[:, ::self.stride, ::self.stride, :]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        K = B.shape[0]

        out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride-1)

        A_grad = conv(out_grad,
                      transpose(flip(B, axes=(0, 1)), (2, 3)),
                      padding=K-self.padding-1)

        B_grad = conv(transpose(A, axes=(3, 0)),
                      transpose(transpose(out_grad, (0, 1)), (1, 2)),
                      padding=self.padding)
        B_grad = transpose(transpose(B_grad, (0, 1)), (1, 2))

        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
