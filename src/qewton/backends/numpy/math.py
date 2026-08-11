from typing import Any
import numpy as np
from qewton.backends.math import MathBackend
from qewton.config.devices import Device, cpu


class NumpyMathBackend(MathBackend[np.ndarray]):
    """NumPy implementation of the MathBackend."""

    add = np.add
    multiply = np.multiply
    subtract = np.subtract
    matmul = np.matmul
    rot90 = np.rot90
    einsum = np.einsum

    # Unary operations
    negative = np.negative
    square = np.square
    sqrt = np.sqrt
    abs = np.abs
    absolute = np.absolute
    fabs = np.fabs
    sign = np.sign

    # Exponential and Logarithmic
    exp = np.exp
    expm1 = np.expm1
    log = np.log
    log10 = np.log10
    log1p = np.log1p
    log2 = np.log2
    logaddexp = np.logaddexp
    logaddexp2 = np.logaddexp2

    # Trigonometric
    sin = np.sin
    cos = np.cos
    tan = np.tan
    arcsin = np.arcsin
    arccos = np.arccos
    arctan = np.arctan
    arctan2 = np.arctan2
    sinh = np.sinh
    cosh = np.cosh
    tanh = np.tanh
    arcsinh = np.arcsinh
    arccosh = np.arccosh
    arctanh = np.arctanh
    hypot = np.hypot
    deg2rad = np.deg2rad
    rad2deg = np.rad2deg

    # Rounding and Comparison
    ceil = np.ceil
    floor = np.floor
    round = np.round
    trunc = np.trunc
    maximum = np.maximum
    minimum = np.minimum
    fmax = np.fmax
    fmin = np.fmin
    clip = np.clip
    real = np.real
    imag = np.imag

    # Logic
    equal = np.equal
    not_equal = np.not_equal
    greater = np.greater
    greater_equal = np.greater_equal
    less = np.less
    less_equal = np.less_equal
    isnan = np.isnan
    isinf = np.isinf
    isfinite = np.isfinite
    isneginf = np.isneginf
    isposinf = np.isposinf
    isreal = np.isreal
    isin = np.isin
    logical_not = np.logical_not
    logical_and = np.logical_and
    logical_or = np.logical_or
    logical_xor = np.logical_xor
    isclose = np.isclose
    allclose = np.allclose
    delete = np.delete

    # Array Manipulation
    reshape = np.reshape
    copy = np.copy
    squeeze = np.squeeze
    ravel = np.ravel
    moveaxis = np.moveaxis
    swapaxes = np.swapaxes
    transpose = np.transpose
    flip = np.flip
    fliplr = np.fliplr
    flipud = np.flipud
    roll = np.roll
    broadcast_to = np.broadcast_to
    concatenate = np.concatenate
    stack = np.stack
    array_split = np.array_split
    vstack = np.vstack
    hstack = np.hstack
    dstack = np.dstack
    hsplit = np.hsplit
    vsplit = np.vsplit
    dsplit = np.dsplit
    expand_dims = np.expand_dims
    append = np.append
    repeat = np.repeat
    tile = np.tile
    pad = np.pad
    nonzero = np.nonzero
    count_nonzero = np.count_nonzero
    split = np.split
    shape = np.shape

    # Linear Algebra
    diag = np.diag
    diagflat = np.diagflat
    diagonal = np.diagonal
    dot = np.dot
    vdot = np.vdot
    inner = np.inner
    outer = np.outer
    kron = np.kron
    tensordot = np.tensordot
    trace = np.trace
    tri = np.tri
    tril = np.tril
    triu = np.triu
    identity = np.identity
    meshgrid = np.meshgrid
    cross = np.cross

    # Others
    diff = np.diff
    take_along_axis = np.take_along_axis
    sort = np.sort
    argsort = np.argsort

    @staticmethod
    def where(condition: Any, x1: Any = None, x2: Any = None) -> np.ndarray:
        if x1 is None and x2 is None:
            return np.where(condition)
        if x1 is not None and x2 is None:
            x2 = 0
        return np.where(condition, x1, x2)

    ptp = np.ptp
    nan_to_num = np.nan_to_num
    digitize = np.digitize
    searchsorted = np.searchsorted
    unique = np.unique
    argpartition = np.argpartition
    unravel_index = np.unravel_index
    corrcoef = np.corrcoef
    correlate = np.correlate
    select = np.select
    slogdet = np.linalg.slogdet
    vectorize = np.vectorize
    power = np.power
    divide = np.divide
    true_divide = np.true_divide
    floor_divide = np.floor_divide
    mod = np.mod
    fmod = np.fmod
    reciprocal = np.reciprocal
    cbrt = np.cbrt
    angle = np.angle
    conj = np.conj
    conjugate = np.conjugate
    i0 = np.i0
    sinc = np.sinc
    gcd = np.gcd
    lcm = np.lcm
    ldexp = np.ldexp
    nextafter = np.nextafter
    signbit = np.signbit
    bitwise_and = np.bitwise_and
    bitwise_or = np.bitwise_or
    bitwise_xor = np.bitwise_xor
    bitwise_not = np.bitwise_not
    bitwise_invert = np.invert
    left_shift = np.left_shift
    right_shift = np.right_shift
    heaviside = np.heaviside
    bartlett = np.bartlett
    hamming = np.hamming
    hanning = np.hanning
    blackman = np.blackman
    kaiser = np.kaiser
    ndim = np.ndim
    size = np.size
    histogram = np.histogram
    cumsum = np.cumsum
    cumprod = np.cumprod
    take = np.take

    # Reduction - wrappers to handle argument mapping
    mean = np.mean
    max = np.max
    min = np.min
    sum = np.sum
    prod = np.prod
    std = np.std
    var = np.var
    all = np.all
    any = np.any
    amax = np.amax
    amin = np.amin
    median = np.median
    argmax = np.argmax
    argmin = np.argmin
    nanargmax = np.nanargmax
    nanargmin = np.nanargmin
    nanmax = np.nanmax
    nanmean = np.nanmean
    nanmedian = np.nanmedian
    nanmin = np.nanmin
    nanprod = np.nanprod
    nanstd = np.nanstd
    nansum = np.nansum
    nanvar = np.nanvar
    nanpercentile = np.nanpercentile
    nanquantile = np.nanquantile
    percentile = np.percentile
    quantile = np.quantile
    average = np.average

    # Factories
    @staticmethod
    def ones(shape: Any, dtype: Any = None, device: Device | str = cpu) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def ones_like(x: Any, dtype: Any = None, device: Device | str = cpu) -> np.ndarray:
        return np.ones_like(x, dtype=dtype)

    @staticmethod
    def zeros(shape: Any, dtype: Any = None, device: Device | str = cpu) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def zeros_like(x: Any, dtype: Any = None, device: Device | str = cpu) -> np.ndarray:
        return np.zeros_like(x, dtype=dtype)

    @staticmethod
    def empty(shape: Any, dtype: Any = None, device: Device | str = cpu) -> np.ndarray:
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def empty_like(x: Any, dtype: Any = None, device: Device | str = cpu) -> np.ndarray:
        return np.empty_like(x, dtype=dtype)

    @staticmethod
    def full(
        shape: Any, fill_value: Any, dtype: Any = None, device: Device | str = cpu
    ) -> np.ndarray:
        return np.full(shape, fill_value, dtype=dtype)

    @staticmethod
    def full_like(
        x: Any, fill_value: Any, dtype: Any = None, device: Device | str = cpu
    ) -> np.ndarray:
        return np.full_like(x, fill_value, dtype=dtype)

    @staticmethod
    def eye(
        N: int,
        M: int | None = None,
        k: int = 0,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> np.ndarray:
        return np.eye(N, M=M, k=k, dtype=dtype)

    @staticmethod
    def linspace(
        start: Any,
        stop: Any,
        num: int = 50,
        endpoint: bool = True,
        retstep: bool = False,
        dtype: Any = None,
        axis: int = 0,
        device: Device | str = cpu,
    ) -> np.ndarray:
        return np.linspace(
            start,
            stop,
            num=num,
            endpoint=endpoint,
            retstep=retstep,
            dtype=dtype,
            axis=axis,
        )

    @staticmethod
    def logspace(
        start: Any,
        stop: Any,
        num: int = 50,
        endpoint: bool = True,
        base: float = 10.0,
        dtype: Any = None,
        axis: int = 0,
        device: Device | str = cpu,
    ) -> np.ndarray:
        return np.logspace(
            start, stop, num=num, endpoint=endpoint, base=base, dtype=dtype, axis=axis
        )

    @staticmethod
    def arange(
        start: Any,
        stop: Any = None,
        step: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> np.ndarray:
        return np.arange(start, stop=stop, step=step, dtype=dtype)

    @staticmethod
    def geomspace(
        start: Any,
        stop: Any,
        num: int = 50,
        endpoint: bool = True,
        dtype: Any = None,
        axis: int = 0,
    ) -> np.ndarray:
        return np.geomspace(
            start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis
        )

    # API mismatch wrappers
    @staticmethod
    def view(x: Any, dtype: Any = None) -> np.ndarray:
        return x.view(dtype=dtype)

    @staticmethod
    def slice(x: Any, slice_config: Any) -> np.ndarray:
        return x[slice_config]

    @staticmethod
    def unsqueeze(x: Any, axis: Any = None) -> np.ndarray:
        return np.expand_dims(x, axis=axis)

    @staticmethod
    def bincount(
        x: Any, weights: Any = None, minlength: int = 0, sparse: bool = False
    ) -> np.ndarray:
        if sparse:
            raise NotImplementedError("Sparse bincount is not supported by NumPyBackend.")
        return np.bincount(x, weights=weights, minlength=minlength)

    @staticmethod
    def divide_no_nan(x1: Any, x2: Any) -> np.ndarray:
        return np.where(x2 != 0, np.divide(x1, x2), 0)

    @staticmethod
    def trapezoid(y: Any, x: Any = None, dx: float = 1.0, axis: int = -1) -> np.ndarray:
        # np.trapezoid was added in 1.25.0, older versions use np.trapz
        func = getattr(np, "trapezoid", getattr(np, "trapz"))
        return func(y, x=x, dx=dx, axis=axis)

    @staticmethod
    def flatten(x: Any, start_dim: int = 0, end_dim: int = -1) -> np.ndarray:
        shape = x.shape
        if end_dim < 0:
            end_dim = len(shape) + end_dim
        if start_dim == 0 and end_dim == len(shape) - 1:
            return x.ravel()
        new_shape = shape[:start_dim] + (-1,) + shape[end_dim + 1 :]
        return x.reshape(new_shape)
