from typing import Annotated

import tensorflow as tf
from qewton.config.dtypes import (
    BFloat16,
    Bool,
    Complex128,
    Complex32,
    Complex64,
    Float16,
    Float32,
    Float64,
    Int16,
    Int32,
    Int64,
    Int8,
    Number,
    UInt16,
    UInt32,
    UInt64,
    UInt8,
)
from qewton.backends.base import DeepLearningBackend
from qewton.backends.tensorflow.grad import TensorflowGradBackend
from qewton.backends.tensorflow.math import TensorflowMathBackend
from qewton.backends.tensorflow.nn import TensorflowNNBackend
from qewton.backends.tensorflow.optim import TensorflowOptimBackend


class TensorflowBackend(DeepLearningBackend[tf.Tensor]):
    math = TensorflowMathBackend
    nn = TensorflowNNBackend
    grad = TensorflowGradBackend
    optim = TensorflowOptimBackend

    default_dtype = tf.Tensor

    dtypes = {
        BFloat16: tf.bfloat16,
        Float16: tf.float16,
        Float32: tf.float32,
        Float64: tf.float64,
        Complex32: tf.complex32,
        Complex64: tf.complex64,
        Complex128: tf.complex128,
        UInt8: tf.uint8,
        UInt16: tf.uint16,
        UInt32: tf.uint32,
        UInt64: tf.uint64,
        Int8: tf.int8,
        Int16: tf.int16,
        Int32: tf.int32,
        Int64: tf.int64,
        Number: None,
        Bool: tf.bool,
    }

    @classmethod
    def to(cls, data, device):
        return data

    @classmethod
    def from_numpy(cls, data, dtype=Float32):
        return tf.convert_to_tensor(data, dtype=cls.dtypes[dtype])

    @classmethod
    def to_numpy(cls, data):
        return data.numpy()
