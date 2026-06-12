from typing import Any
import tensorflow as tf
from qewton.backends.math import MathBackend
from qewton.backends.base import TensorType

# TODO: check and test, currently only AI-generated matching


class TensorflowMathBackend(MathBackend[tf.Tensor]):
    # Arithmetic Operations
    add = tf.math.add
    subtract = tf.math.subtract
    multiply = tf.math.multiply
    divide = tf.math.divide
    true_divide = tf.math.truediv
    floor_divide = tf.math.floordiv
    power = tf.math.pow
    mod = tf.math.mod
    fmod = (
        tf.math.floormod
    )  # tf.math.floormod is equivalent to fmod for positive divisors
    negative = tf.math.negative
    square = tf.math.square
    sqrt = tf.math.sqrt
    abs = tf.math.abs
    absolute = tf.math.abs
    fabs = tf.math.abs

    # Exponential and Logarithmic
    exp = tf.math.exp
    exp2 = tf.math.exp2
    expm1 = tf.math.expm1
    log = tf.math.log

    @staticmethod
    def log2(x: Any) -> tf.Tensor:
        return tf.math.log(x) / tf.math.log(tf.constant(2.0, dtype=x.dtype))

    @staticmethod
    def log10(x: Any) -> tf.Tensor:
        return tf.math.log(x) / tf.math.log(tf.constant(10.0, dtype=x.dtype))

    log1p = tf.math.log1p

    # Trigonometric
    sin = tf.math.sin
    cos = tf.math.cos
    tan = tf.math.tan
    arcsin = tf.math.asin
    arccos = tf.math.acos
    arctan = tf.math.atan
    arctan2 = tf.math.atan2
    sinh = tf.math.sinh
    cosh = tf.math.cosh
    tanh = tf.math.tanh
    arcsinh = tf.math.asinh
    arccosh = tf.math.acosh
    arctanh = tf.math.atanh

    # Rounding and Comparison
    ceil = tf.math.ceil
    floor = tf.math.floor
    round = tf.math.round
    trunc = tf.math.trunc
    maximum = tf.math.maximum
    minimum = tf.math.minimum
    fmax = tf.math.maximum  # tf.math.maximum handles NaNs like fmax
    fmin = tf.math.minimum  # tf.math.minimum handles NaNs like fmin
    clip = tf.clip_by_value
    where = tf.where
    equal = tf.math.equal
    not_equal = tf.math.not_equal
    greater = tf.math.greater
    greater_equal = tf.math.greater_equal
    less = tf.math.less
    less_equal = tf.math.less_equal

    # Reductions
    @staticmethod
    def sum(x: Any, axis: Any = None, keepdims: bool = False) -> tf.Tensor:
        return tf.reduce_sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def mean(x: Any, axis: Any = None, keepdims: bool = False) -> tf.Tensor:
        return tf.reduce_mean(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def prod(
        x: Any, axis: Any = None, keepdims: bool = False, dtype: Any = None
    ) -> tf.Tensor:
        return tf.reduce_prod(
            x, axis=axis, keepdims=keepdims
        )  # TF reduce_prod doesn't have dtype arg directly, usually handled by input dtype

    @staticmethod
    def std(x: Any, axis: Any = None, keepdims: bool = False) -> tf.Tensor:
        # TensorFlow doesn't have a direct tf.reduce_std. It can be calculated from variance.
        # tf.math.reduce_std is available in newer TF versions.
        # For broader compatibility, let's implement it using variance.
        if axis is None:
            mean = tf.reduce_mean(x)
            variance = tf.reduce_mean(tf.math.square(x - mean))
        else:
            mean = tf.reduce_mean(x, axis=axis, keepdims=True)
            variance = tf.reduce_mean(
                tf.math.square(x - mean), axis=axis, keepdims=keepdims
            )
        return tf.math.sqrt(variance)

    @staticmethod
    def var(x: Any, axis: Any = None, keepdims: bool = False) -> tf.Tensor:
        if axis is None:
            mean = tf.reduce_mean(x)
            return tf.reduce_mean(tf.math.square(x - mean))
        else:
            mean = tf.reduce_mean(x, axis=axis, keepdims=True)
            return tf.reduce_mean(tf.math.square(x - mean), axis=axis, keepdims=keepdims)

    all = tf.reduce_all
    any = tf.reduce_any
    amax = tf.reduce_max
    amin = tf.reduce_min
    max = tf.reduce_max  # tf.reduce_max handles initial value implicitly by reduction
    min = tf.reduce_min  # tf.reduce_min handles initial value implicitly by reduction

    # Matrix Operations
    matmul = tf.linalg.matmul
    dot = tf.tensordot  # tf.tensordot can act as dot product
    vdot = tf.tensordot  # tf.tensordot can act as vdot product
    inner = tf.einsum  # tf.einsum can implement inner product
    outer = tf.tensordot  # tf.tensordot can implement outer product
    kron = tf.experimental.numpy.kron  # tf.experimental.numpy.kron
    tensordot = tf.tensordot
    einsum = tf.einsum

    # Factory Methods (assuming these are already handled or will be added)
    zeros = tf.zeros
    ones = tf.ones
    empty = tf.empty
    full = tf.fill
    eye = tf.eye
    zeros_like = tf.zeros_like
    ones_like = tf.ones_like
    full_like = tf.fill
    arange = tf.range
    linspace = tf.linspace
    logspace = tf.math.logspace

    # Array Manipulation
    reshape = tf.reshape

    @staticmethod
    def flatten(x: Any, start_dim: int = 0, end_dim: int = -1) -> tf.Tensor:
        shape = tf.shape(x)
        num_dims = tf.shape(shape)[0]

        if end_dim < 0:
            end_dim = num_dims + end_dim

        if start_dim < 0:
            start_dim = num_dims + start_dim

        if start_dim == 0 and end_dim == num_dims - 1:
            return tf.reshape(x, [-1])

        # Calculate the product of dimensions to be flattened
        flattened_dim_size = tf.reduce_prod(shape[start_dim : end_dim + 1])

        # Construct the new shape
        new_shape_list = []
        if start_dim > 0:
            new_shape_list.append(shape[:start_dim])
        new_shape_list.append(tf.expand_dims(flattened_dim_size, axis=0))
        if end_dim < num_dims - 1:
            new_shape_list.append(shape[end_dim + 1 :])

        new_shape = tf.concat(new_shape_list, axis=0)
        return tf.reshape(x, new_shape)

    squeeze = tf.squeeze
    ravel = tf.reshape  # tf.reshape with shape [-1] can act as ravel
    moveaxis = tf.experimental.numpy.moveaxis  # tf.experimental.numpy.moveaxis
    swapaxes = tf.transpose
    flip = tf.reverse
    roll = tf.roll
    rot90 = tf.image.rot90
    tile = tf.tile
    repeat = tf.repeat
    broadcast_to = tf.broadcast_to
    concatenate = tf.concat
    stack = tf.stack
    vstack = tf.stack  # tf.stack with axis=0 can act as vstack
    hstack = tf.stack  # tf.stack with axis=1 can act as hstack
    dstack = tf.stack  # tf.stack with axis=2 can act as dstack

    @staticmethod
    def slice(x: Any, slice_config: Any) -> tf.Tensor:
        return x[slice_config]

    @staticmethod
    def expand_dims(x: Any, axis: Any) -> tf.Tensor:
        return tf.expand_dims(x, axis=axis)

    @staticmethod
    def transpose(x: Any, axes: Any = None) -> tf.Tensor:
        if axes is None:
            # Default transpose for 2D, or reverse dims for >2D
            return tf.transpose(x)
        return tf.transpose(x, perm=axes)

    # Other Utility
    @staticmethod
    def sort(x: Any, axis: int = -1) -> tf.Tensor:
        return tf.sort(x, axis=axis)

    @staticmethod
    def argsort(x: Any, axis: int = -1) -> tf.Tensor:
        return tf.argsort(x, axis=axis)

    @staticmethod
    def argmax(x: Any, axis: Any = None, keepdims: bool = False) -> tf.Tensor:
        return tf.argmax(
            x, axis=axis, output_type=tf.int64
        )  # tf.argmax requires output_type

    @staticmethod
    def argmin(x: Any, axis: Any = None, keepdims: bool = False) -> tf.Tensor:
        return tf.argmin(
            x, axis=axis, output_type=tf.int64
        )  # tf.argmin requires output_type
