from abc import abstractmethod
from typing import Any

from qewton.backends.base import Backend, TensorType
from qewton.config.devices import cpu, Device


class MathBackend(Backend[TensorType]):
    """A Backend that implements all basic mathematical methods,
    method selection inspired by numpy.

    The overall idea is that every math backend implements the numpy methods,
    and mimics numpy's behaviour to unify the usage among all backends.
    """

    # TODO: check these, currently only generated stuff

    @staticmethod
    @abstractmethod
    def add(x1: Any, x2: Any, /) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def multiply(x1: Any, x2: Any, /) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def rot90(array: Any, k: int = 1, axes: tuple[int, int] = (0, 1)) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def einsum(subscripts: str, *operands: Any, **kwargs: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def subtract(x1: Any, x2: Any, /) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def matmul(x1: Any, x2: Any, /) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def mean(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def max(
        x: Any, axis: Any = None, keepdims: bool = False, initial: Any = None
    ) -> TensorType:
        pass

    # Factory methods:

    @staticmethod
    @abstractmethod
    def ones(shape: Any, dtype: Any = None, device: Device = cpu) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def ones_like(x: Any, dtype: Any = None, device: Device = cpu) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def zeros(shape: Any, dtype: Any = None, device: Device = cpu) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def zeros_like(x: Any, dtype: Any = None, device: Device = cpu) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def empty(shape: Any, dtype: Any = None, device: Device = cpu) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def empty_like(x: Any, dtype: Any = None, device: Device = cpu) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def eye(
        N: int, M: int | None = None, k: int = 0, dtype: Any = None, device: Device = cpu
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def linspace(
        start: Any,
        stop: Any,
        num: int = 50,
        endpoint: bool = True,
        retstep: bool = False,
        dtype: Any = None,
        axis: int = 0,
        device: Device = cpu,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def logspace(
        start: Any,
        stop: Any,
        num: int = 50,
        endpoint: bool = True,
        base: float = 10.0,
        dtype: Any = None,
        axis: int = 0,
        device: Device = cpu,
    ) -> TensorType:
        pass

    ########

    @staticmethod
    @abstractmethod
    def absolute(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def abs(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def fabs(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def all(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def angle(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def any(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def amax(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def amin(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def append(x1: Any, x2: Any, axis: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def arange(
        start: Any,
        stop: Any = None,
        step: Any = None,
        dtype: Any = None,
        device: Device = cpu,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def arccos(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def arccosh(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def arcsin(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def arcsinh(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def arctan(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def arctan2(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def arctanh(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def argmax(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def argmin(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def argsort(x: Any, axis: int = -1) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def array(x: Any, dtype: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def view(x: Any, dtype: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def average(x: Any, axis: Any = None, weights: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def bartlett(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def hamming(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def hanning(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def heaviside(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def kaiser(x: Any, beta: float) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def bincount(
        x: Any, weights: Any = None, minlength: int = 0, sparse: bool = False
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def bitwise_and(x: Any, y: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def bitwise_invert(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def bitwise_not(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def bitwise_or(x: Any, y: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def bitwise_xor(x: Any, y: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def bitwise_left_shift(x: Any, y: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def left_shift(x: Any, y: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def bitwise_right_shift(x: Any, y: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def right_shift(x: Any, y: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def blackman(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def broadcast_to(x: Any, shape: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def cbrt(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def ceil(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def clip(x: Any, x_min: Any, x_max: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def concatenate(xs: Any, axis: int = 0) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def conjugate(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def conj(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def copy(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def cos(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def cosh(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def count_nonzero(x: Any, axis: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def cross(
        x1: Any,
        x2: Any,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: Any = None,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def cumprod(x: Any, axis: Any = None, dtype: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def cumsum(x: Any, axis: Any = None, dtype: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def deg2rad(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def rad2deg(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def diag(x: Any, k: int = 0) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def diagflat(x: Any, k: int = 0) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def diagonal(x: Any, offset: int = 0, axis1: int = 0, axis2: int = 1) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def diff(a: Any, n: int = 1, axis: int = -1) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def digitize(x: Any, bins: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def dot(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def dstack(xs: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def equal(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def exp(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def exp2(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def expand_dims(x: Any, axis: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def slice(x: Any, slice_config: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def expm1(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def flip(x: Any, axis: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def fliplr(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def flipud(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def floor(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def full(
        shape: Any, fill_value: Any, dtype: Any = None, device: Device = cpu
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def full_like(
        x: Any, fill_value: Any, dtype: Any = None, device: Device = cpu
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def gcd(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def geomspace(
        start: Any,
        stop: Any,
        num: int = 50,
        endpoint: bool = True,
        dtype: Any = None,
        axis: int = 0,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def greater(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def greater_equal(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def hstack(xs: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def hsplit(x: Any, indices_or_sections: Any) -> list[TensorType]:
        pass

    @staticmethod
    @abstractmethod
    def hypot(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def identity(n: int, dtype: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def imag(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def i0(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def isclose(
        x1: Any, x2: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def allclose(
        x1: Any, x2: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def isfinite(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def isin(
        x1: Any, x2: Any, assume_unique: bool = False, invert: bool = False
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def isinf(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def isnan(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def isneginf(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def isposinf(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def isreal(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def kron(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def lcm(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def ldexp(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def less(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def less_equal(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def log(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def log10(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def log1p(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def log2(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def logaddexp(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def logaddexp2(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def logical_and(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def logical_not(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def logical_or(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def maximum(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def fmax(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def median(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def meshgrid(*x: Any, indexing: str = "xy") -> list[TensorType]:
        pass

    @staticmethod
    @abstractmethod
    def min(
        x: Any, axis: Any = None, keepdims: bool = False, initial: Any = None
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def minimum(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def fmin(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def mod(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def fmod(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def moveaxis(x: Any, source: Any, destination: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nanargmax(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nanargmin(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nancumsum(x: Any, axis: Any = None, dtype: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nancumprod(x: Any, axis: Any = None, dtype: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nanmax(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nanmean(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nanmedian(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nanmin(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nanpercentile(
        x: Any, q: Any, axis: Any = None, method: str = "linear", keepdims: bool = False
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nanprod(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nanquantile(
        x: Any, q: Any, axis: Any = None, method: str = "linear", keepdims: bool = False
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nanstd(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nansum(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nanvar(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nan_to_num(
        x: Any, nan: float = 0.0, posinf: Any = None, neginf: Any = None
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def ndim(x: Any) -> int:
        pass

    @staticmethod
    @abstractmethod
    def nonzero(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def not_equal(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def outer(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def pad(
        x: Any, pad_width: Any, mode: str = "constant", constant_values: Any = None
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def percentile(
        x: Any, q: Any, axis: Any = None, method: str = "linear", keepdims: bool = False
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def prod(
        x: Any, axis: Any = None, keepdims: bool = False, dtype: Any = None
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def ptp(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def quantile(
        x: Any, q: Any, axis: Any = None, method: str = "linear", keepdims: bool = False
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def ravel(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def unravel_index(indices: Any, shape: Any) -> tuple[TensorType, ...]:
        pass

    @staticmethod
    @abstractmethod
    def real(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def reciprocal(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def repeat(x: Any, repeats: Any, axis: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def reshape(x: Any, newshape: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def roll(x: Any, shift: Any, axis: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def searchsorted(sorted_sequence: Any, values: Any, side: str = "left") -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def sign(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def signbit(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def sin(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def sinc(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def sinh(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def size(x: Any) -> int:
        pass

    @staticmethod
    @abstractmethod
    def sort(x: Any, axis: int = -1) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def split(x: Any, indices_or_sections: Any, axis: int = 0) -> list[TensorType]:
        pass

    @staticmethod
    @abstractmethod
    def array_split(x: Any, indices_or_sections: Any, axis: int = 0) -> list[TensorType]:
        pass

    @staticmethod
    @abstractmethod
    def stack(x: list[Any], axis: int = 0) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def std(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def swapaxes(x: Any, axis1: int, axis2: int) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def take(x: Any, indices: Any, axis: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def take_along_axis(x: Any, indices: Any, axis: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def tan(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def tanh(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def tensordot(x1: Any, x2: Any, axes: Any = 2) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def round(x: Any, decimals: int = 0) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def tile(x: Any, repeats: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def trace(x: Any, offset: int = 0, axis1: int = 0, axis2: int = 1) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def tri(N: int, M: int | None = None, k: int = 0, dtype: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def tril(x: Any, k: int = 0) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def triu(x: Any, k: int = 0) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def trunc(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def vdot(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def inner(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def vstack(xs: list[Any]) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def vsplit(x: Any, indices_or_sections: Any) -> list[TensorType]:
        pass

    @staticmethod
    @abstractmethod
    def vectorize(pyfunc: Any, *, excluded: Any = None, signature: Any = None) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def where(condition: Any, x1: Any = None, x2: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def divide(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def divide_no_nan(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def true_divide(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def power(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def negative(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def nextafter(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def square(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def sqrt(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def squeeze(x: Any, axis: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def unsqueeze(x: Any, axis: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def transpose(x: Any, axes: Any = None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def trapezoid(y: Any, x: Any = None, dx: float = 1.0, axis: int = -1) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def vander(x: Any, N: int | None = None, increasing: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def var(x: Any, axis: Any = None, keepdims: bool = False) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def sum(
        x: Any, axis: Any = None, keepdims: bool = False
    ) -> TensorType:  # Removed dtype for consistency
        pass

    @staticmethod
    @abstractmethod
    def floor_divide(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def logical_xor(x1: Any, x2: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def corrcoef(x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def correlate(x1: Any, x2: Any, mode: str = "valid") -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def select(
        condlist: list[Any], choicelist: list[Any], default: Any = 0
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def slogdet(x: Any) -> tuple[TensorType, TensorType]:
        pass

    @staticmethod
    @abstractmethod
    def argpartition(x: Any, kth: int, axis: int = -1) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def histogram(
        x: Any, bins: int = 10, range: Any = None
    ) -> tuple[TensorType, TensorType]:
        pass

    @staticmethod
    @abstractmethod
    def unique(
        x: Any,
        sorted: bool = True,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
        axis: int | None = None,
        size: int | None = None,
        fill_value: Any = None,
    ) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def dsplit(x: Any, indices_or_sections: Any) -> list[TensorType]:
        pass

    @staticmethod
    @abstractmethod
    def flatten(x: Any, start_dim: int = 0, end_dim: int = -1) -> TensorType:
        pass
