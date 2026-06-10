from typing import Any
import torch
from qewton.backends.math import MathBackend
from qewton.backends.base import TensorType


class TorchMathBackend(MathBackend[torch.Tensor]):
    # Arithmetic Operations (Direct assignment works for positional-only)
    add = torch.add
    subtract = torch.subtract
    multiply = torch.multiply
    divide = torch.divide
    true_divide = torch.true_divide
    floor_divide = torch.floor_divide
    power = torch.pow
    mod = torch.remainder
    fmod = torch.fmod
    negative = torch.neg
    square = torch.square
    sqrt = torch.sqrt
    abs = torch.abs
    absolute = torch.abs
    fabs = torch.abs

    # Exponential and Logarithmic
    exp = torch.exp
    exp2 = torch.exp2
    expm1 = torch.expm1
    log = torch.log
    log2 = torch.log2
    log10 = torch.log10
    log1p = torch.log1p

    # Trigonometric
    sin = torch.sin
    cos = torch.cos
    tan = torch.tan
    arcsin = torch.asin
    arccos = torch.acos
    arctan = torch.atan
    arctan2 = torch.atan2
    sinh = torch.sinh
    cosh = torch.cosh
    tanh = torch.tanh
    arcsinh = torch.asinh
    arccosh = torch.acosh
    arctanh = torch.atanh

    # Rounding and Comparison
    ceil = torch.ceil
    floor = torch.floor
    round = torch.round
    trunc = torch.trunc
    maximum = torch.maximum
    minimum = torch.minimum
    fmax = torch.fmax
    fmin = torch.fmin
    clip = torch.clamp
    where = torch.where
    equal = torch.eq
    not_equal = torch.ne
    greater = torch.gt
    greater_equal = torch.ge
    less = torch.lt
    less_equal = torch.le

    # Reductions (Translating keyword names axis -> dim, keepdims -> keepdim)
    @staticmethod
    def sum(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.sum(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def mean(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.mean(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def prod(
        x: Any, axis: Any = None, keepdims: bool = False, dtype: Any = None
    ) -> torch.Tensor:
        return torch.prod(x, dim=axis, keepdim=keepdims, dtype=dtype)

    @staticmethod
    def std(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.std(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def var(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.var(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def all(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.all(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def any(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.any(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def amax(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.amax(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def amin(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.amin(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def max(
        x: Any, axis: Any = None, keepdims: bool = False, initial: Any = None
    ) -> torch.Tensor:
        # Using amax for reduction to match NumPy/Keras behavior
        res = torch.amax(x, dim=axis, keepdim=keepdims)
        return (
            torch.maximum(res, torch.tensor(initial, device=res.device))
            if initial is not None
            else res
        )

    @staticmethod
    def min(
        x: Any, axis: Any = None, keepdims: bool = False, initial: Any = None
    ) -> torch.Tensor:
        res = torch.amin(x, dim=axis, keepdim=keepdims)
        return (
            torch.minimum(res, torch.tensor(initial, device=res.device))
            if initial is not None
            else res
        )

    # Matrix Operations
    matmul = torch.matmul
    dot = torch.dot
    vdot = torch.vdot
    inner = torch.inner
    outer = torch.outer
    kron = torch.kron
    tensordot = torch.tensordot
    einsum = torch.einsum

    # Factory Methods
    zeros = torch.zeros
    ones = torch.ones
    empty = torch.empty
    full = torch.full
    eye = torch.eye
    zeros_like = torch.zeros_like
    ones_like = torch.ones_like
    full_like = torch.full_like
    arange = torch.arange
    linspace = torch.linspace
    logspace = torch.logspace

    # Array Manipulation
    reshape = torch.reshape
    squeeze = torch.squeeze
    ravel = torch.ravel
    moveaxis = torch.moveaxis
    swapaxes = torch.swapaxes
    flip = torch.flip
    roll = torch.roll
    rot90 = torch.rot90
    tile = torch.tile
    repeat = torch.repeat_interleave
    broadcast_to = torch.broadcast_to
    concatenate = torch.cat
    stack = torch.stack
    vstack = torch.vstack
    hstack = torch.hstack
    dstack = torch.dstack

    @staticmethod
    def expand_dims(x: Any, axis: Any) -> torch.Tensor:
        return torch.unsqueeze(x, dim=axis)

    @staticmethod
    def transpose(x: Any, axes: Any = None) -> torch.Tensor:
        if axes is None:
            return x.T
        return torch.permute(x, dims=axes)

    # Other Utility
    @staticmethod
    def sort(x: Any, axis: int = -1) -> torch.Tensor:
        return torch.sort(x, dim=axis).values

    @staticmethod
    def argsort(x: Any, axis: int = -1) -> torch.Tensor:
        return torch.argsort(x, dim=axis)

    @staticmethod
    def argmax(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.argmax(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def argmin(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.argmin(x, dim=axis, keepdim=keepdims)
