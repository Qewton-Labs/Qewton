from typing import Any
import torch
import math
from qewton.backends.math import MathBackend
from qewton.backends.torch.device import get_torch_device
from qewton.config.devices import Device, cpu


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
    # Unary operations
    negative = torch.neg
    square = torch.square
    sqrt = torch.sqrt
    abs = torch.abs
    absolute = torch.abs
    fabs = torch.abs
    sign = torch.sign

    # Exponential and Logarithmic
    exp = torch.exp
    exp2 = torch.exp2
    expm1 = torch.expm1
    # Logarithmic
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
    hypot = torch.hypot
    deg2rad = torch.deg2rad
    rad2deg = torch.rad2deg

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
    real = torch.real

    @staticmethod
    def count_nonzero(x: Any, axis: Any = None) -> torch.Tensor:
        return torch.count_nonzero(x, dim=axis)

    @staticmethod
    def nonzero(x: Any) -> torch.Tensor:
        return torch.nonzero(x, as_tuple=True)

    @staticmethod
    def where(condition: Any, x1: Any = None, x2: Any = None) -> torch.Tensor:
        if x1 is None and x2 is None:
            return torch.where(condition)
        if x1 is not None and x2 is None:
            x2 = 0
        return torch.where(condition, x1, x2)

    @staticmethod
    def ptp(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            return torch.max(x) - torch.min(x)
        return torch.amax(x, dim=axis, keepdim=keepdims) - torch.amin(
            x, dim=axis, keepdim=keepdims
        )

    equal = torch.eq
    not_equal = torch.ne
    greater = torch.gt
    greater_equal = torch.ge
    less = torch.lt
    less_equal = torch.le
    isnan = torch.isnan
    isinf = torch.isinf
    isfinite = torch.isfinite
    nan_to_num = torch.nan_to_num
    nanmedian = torch.nanmedian

    @staticmethod
    def median(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            return torch.median(x)
        return torch.median(x, dim=axis, keepdim=keepdims).values

    @staticmethod
    def nansum(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.nansum(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def nanmean(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            return torch.nanmean(x)
        return torch.nanmean(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def nanmin(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        mask = torch.isnan(x)

        if axis is None:
            if torch.all(mask):
                raise ValueError("All-NaN slice encountered")

            return torch.min(torch.nan_to_num(x, nan=float("inf")))

        if torch.any(torch.all(mask, dim=axis)):
            raise ValueError("All-NaN slice encountered")

        return torch.min(
            torch.nan_to_num(x, nan=float("inf")), dim=axis, keepdim=keepdims
        ).values

    @staticmethod
    def nanmax(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        mask = torch.isnan(x)

        if axis is None:
            if torch.all(mask):
                raise ValueError("All-NaN slice encountered")

            return torch.max(torch.nan_to_num(x, nan=-float("inf")))

        if torch.any(torch.all(mask, dim=axis)):
            raise ValueError("All-NaN slice encountered")

        return torch.max(
            torch.nan_to_num(x, nan=-float("inf")), dim=axis, keepdim=keepdims
        ).values

    @staticmethod
    def nanargmax(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        mask = torch.isnan(x)

        if axis is None:
            if torch.all(mask):
                raise ValueError("All-NaN slice encountered")

            return torch.argmax(torch.nan_to_num(x, nan=-float("inf")))

        if torch.any(torch.all(mask, dim=axis)):
            raise ValueError("All-NaN slice encountered")

        return torch.argmax(
            torch.nan_to_num(x, nan=-float("inf")), dim=axis, keepdim=keepdims
        )

    @staticmethod
    def nanargmin(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        mask = torch.isnan(x)

        if axis is None:
            if torch.all(mask):
                raise ValueError("All-NaN slice encountered")

            return torch.argmin(torch.nan_to_num(x, nan=float("inf")))

        if torch.any(torch.all(mask, dim=axis)):
            raise ValueError("All-NaN slice encountered")

        return torch.argmin(
            torch.nan_to_num(x, nan=float("inf")), dim=axis, keepdim=keepdims
        )

    @staticmethod
    def isclose(
        x1: Any,
        x2: Any,
        rtol: float = 0.00001,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> torch.Tensor:
        return torch.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)

    @staticmethod
    def allclose(
        x1: Any,
        x2: Any,
        rtol: float = 0.00001,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> torch.Tensor:
        return torch.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)

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
        if axis is None:
            return torch.prod(x, dtype=dtype)
        return torch.prod(x, dim=axis, keepdim=keepdims, dtype=dtype)

    @staticmethod
    def std(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.std(x, dim=axis, keepdim=keepdims)

    @staticmethod
    def var(x: Any, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return torch.var(x, dim=axis, keepdim=keepdims, correction=0)

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
    identity = torch.eye

    @staticmethod
    def diag(x: Any, k: int = 0) -> torch.Tensor:
        return torch.diag(x, diagonal=k)

    diagonal = torch.diagonal
    matmul = torch.matmul
    dot = torch.inner

    @staticmethod
    def vdot(x: Any, y: Any) -> torch.Tensor:
        return torch.sum(torch.conj(x).reshape(-1) * y.reshape(-1))

    inner = torch.inner
    diff = torch.diff
    outer = torch.outer
    kron = torch.kron

    @staticmethod
    def tensordot(x1: Any, x2: Any, axes: Any = 2) -> torch.Tensor:
        return torch.tensordot(x1, x2, dims=axes)

    einsum = torch.einsum

    @staticmethod
    def cumsum(x: Any, axis: Any = None, dtype: Any = None) -> torch.Tensor:
        if axis is None:
            axis = 0
        return torch.cumsum(x, dim=axis, dtype=dtype)

    @staticmethod
    def cumprod(x: Any, axis: Any = None, dtype: Any = None) -> torch.Tensor:
        if axis is None:
            axis = 0
        return torch.cumprod(x, dim=axis, dtype=dtype)

    @staticmethod
    def take(x: Any, indices: Any, axis: Any = None) -> torch.Tensor:
        if axis is None:
            return torch.take(x, torch.tensor(indices))
        return torch.index_select(x, axis, torch.tensor(indices))

    @staticmethod
    def triu(x: Any, k: int = 0) -> torch.Tensor:
        return torch.triu(x, diagonal=k)

    @staticmethod
    def tril(x: Any, k: int = 0) -> torch.Tensor:
        return torch.tril(x, diagonal=k)

    @staticmethod
    def trace(x: Any, offset: int = 0, axis1: int = 0, axis2: int = 1) -> torch.Tensor:
        return torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2).sum(dim=-1)

    # Factory Methods
    meshgrid = torch.meshgrid

    @staticmethod
    def zeros(shape: Any, dtype: Any = None, device: Device = cpu) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=get_torch_device(device))

    @staticmethod
    def ones(shape: Any, dtype: Any = None, device: Device = cpu) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype, device=get_torch_device(device))

    @staticmethod
    def zeros_like(x: Any, dtype: Any = None, device: Device = cpu) -> torch.Tensor:
        return torch.zeros_like(x, dtype=dtype, device=get_torch_device(device))

    @staticmethod
    def ones_like(x: Any, dtype: Any = None, device: Device = cpu) -> torch.Tensor:
        return torch.ones_like(x, dtype=dtype, device=get_torch_device(device))

    @staticmethod
    def empty(shape: Any, dtype: Any = None, device: Device = cpu) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=get_torch_device(device))

    @staticmethod
    def empty_like(x: Any, dtype: Any = None, device: Device = cpu) -> torch.Tensor:
        return torch.empty_like(x, dtype=dtype, device=get_torch_device(device))

    @staticmethod
    def full(
        shape: Any, fill_value: Any, dtype: Any = None, device: Device = cpu
    ) -> torch.Tensor:
        return torch.full(shape, fill_value, dtype=dtype, device=get_torch_device(device))

    @staticmethod
    def full_like(
        x: Any, fill_value: Any, dtype: Any = None, device: Device = cpu
    ) -> torch.Tensor:
        return torch.full_like(
            x, fill_value, dtype=dtype, device=get_torch_device(device)
        )

    @staticmethod
    def eye(
        N: int, M: int | None = None, k: int = 0, dtype: Any = None, device: Device = cpu
    ) -> torch.Tensor:
        m_val = M if M is not None else N
        res = torch.zeros((N, m_val), dtype=dtype, device=get_torch_device(device))
        if k >= 0:
            row_idx = torch.arange(0, min(N, m_val - k), device=res.device)
            col_idx = row_idx + k
        else:
            col_idx = torch.arange(0, min(m_val, N + k), device=res.device)
            row_idx = col_idx - k
        if len(row_idx) > 0:
            res[row_idx, col_idx] = 1
        return res

    @staticmethod
    def arange(
        start: Any,
        stop: Any = None,
        step: Any = None,
        dtype: Any = None,
        device: Device = cpu,
    ) -> torch.Tensor:
        if stop is None:
            return torch.arange(start, dtype=dtype, device=get_torch_device(device))
        if step is None:
            return torch.arange(start, stop, dtype=dtype, device=get_torch_device(device))
        return torch.arange(
            start, stop, step, dtype=dtype, device=get_torch_device(device)
        )

    @staticmethod
    def linspace(
        start: Any,
        stop: Any,
        num: int = 50,
        endpoint: bool = True,
        retstep: bool = False,
        dtype: Any = None,
        axis: int = 0,
        device: Device = cpu,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        device_torch = get_torch_device(device)
        if not endpoint and num > 0:
            step = (stop - start) / num
            res = torch.linspace(
                start, stop - step, num, dtype=dtype, device=device_torch
            )
        else:
            res = torch.linspace(start, stop, num, dtype=dtype, device=device_torch)
        if retstep:
            actual_step = (
                (stop - start) / (num - 1)
                if endpoint and num > 1
                else (stop - start) / num
            )
            return res, torch.tensor(actual_step, device=device_torch)
        return res

    @staticmethod
    def logspace(
        start: Any,
        stop: Any,
        num: int = 50,
        endpoint: bool = True,
        base: float = 10.0,
        dtype: Any = None,
        axis: int = 0,
        device: Device = cpu,
    ) -> torch.Tensor:
        device_torch = get_torch_device(device)
        if not endpoint and num > 0:
            step = (stop - start) / num
            return torch.logspace(
                start, stop - step, num, base=base, dtype=dtype, device=device_torch
            )
        return torch.logspace(
            start, stop, num, base=base, dtype=dtype, device=device_torch
        )

    # Array Manipulation
    @staticmethod
    def reshape(x: Any, newshape: Any) -> torch.Tensor:
        if isinstance(newshape, int):
            return torch.reshape(x, (newshape,))
        return torch.reshape(x, newshape)

    @staticmethod
    def flatten(x: Any, start_dim: int = 0, end_dim: int = -1) -> torch.Tensor:
        return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)

    copy = torch.clone
    squeeze = torch.squeeze
    unsqueeze = torch.unsqueeze
    ravel = torch.ravel
    moveaxis = torch.moveaxis
    swapaxes = torch.swapaxes

    @staticmethod
    def append(x1: Any, x2: Any, axis: Any = None) -> torch.Tensor:
        if axis is None:
            axis = 0
        return torch.concat([x1, x2], dim=axis)

    @staticmethod
    def flip(x: Any, axis: Any = None) -> torch.Tensor:
        if isinstance(axis, int):
            axis = (axis,)
        if axis is None:
            axis = (0,)
        return torch.flip(x, dims=axis)

    @staticmethod
    def roll(x: Any, shift: Any, axis: Any = None) -> torch.Tensor:
        return torch.roll(x, shifts=shift, dims=axis)

    rot90 = torch.rot90

    @staticmethod
    def size(x: Any) -> int:
        return math.prod(x.shape)

    @staticmethod
    def ndim(x: Any) -> int:
        return len(x.shape)

    @staticmethod
    def tile(x: Any, repeats: Any) -> torch.Tensor:
        if isinstance(repeats, int):
            return torch.tile(x, (repeats,))
        return torch.tile(x, dims=repeats)

    repeat = torch.repeat_interleave
    broadcast_to = torch.broadcast_to

    @staticmethod
    def concatenate(tensors: Any, axis: int = 0) -> torch.Tensor:
        return torch.cat(tensors, dim=axis)

    @staticmethod
    def stack(tensors: Any, axis: int = 0) -> torch.Tensor:
        return torch.stack(tensors, dim=axis)

    @staticmethod
    def split(
        x: Any, split_size_or_sections: Any, axis: int = 0
    ) -> tuple[torch.Tensor, ...]:
        return torch.split(x, split_size_or_sections, dim=axis)

    vstack = torch.vstack
    hstack = torch.hstack
    dstack = torch.dstack
    reciprocal = torch.reciprocal

    @staticmethod
    def slice(x: Any, slice_config: Any) -> torch.Tensor:
        return x[slice_config]

    @staticmethod
    def expand_dims(x: Any, axis: Any) -> torch.Tensor:
        return torch.unsqueeze(x, dim=axis)

    @staticmethod
    def transpose(x: Any, axes: Any = None) -> torch.Tensor:
        if axes is None:
            return x.T
        return torch.permute(x, dims=axes)

    @staticmethod
    def pad(
        x: Any, pad_width: Any, mode: str = "constant", constant_values: Any = None
    ) -> torch.Tensor:
        pad = []

        for before, after in reversed(pad_width):
            pad.extend([before, after])

        torch_mode = {
            "constant": "constant",
            "reflect": "reflect",
            "edge": "replicate",
            "wrap": "circular",
        }[mode]

        return torch.nn.functional.pad(
            x, tuple(pad), mode=torch_mode, value=constant_values
        )

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

    # Logic methods:
    logical_not = torch.logical_not
    logical_and = torch.logical_and
    logical_or = torch.logical_or
    logical_xor = torch.logical_xor
