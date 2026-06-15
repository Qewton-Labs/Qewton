import numpy as np

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

from qewton.backends.base import Backend
from qewton.backends.numpy.math import NumpyMathBackend


class NumPyBackend(Backend[np.ndarray]):
    default_dtype = np.ndarray

    math = NumpyMathBackend

    dtypes = {
        BFloat16: False,
        Float16: np.float16,
        Float32: np.float32,
        Float64: np.float64,
        Complex32: False,
        Complex64: np.complex64,
        Complex128: np.complex128,
        UInt8: np.uint8,
        UInt16: np.uint16,
        UInt32: np.uint32,
        UInt64: np.uint64,
        Int8: np.int8,
        Int16: np.int16,
        Int32: np.int32,
        Int64: np.int64,
        Number: None,
        Bool: np.bool,
    }

    @staticmethod
    def load(path, **kwargs):
        return np.load(path, **kwargs)

    @classmethod
    def from_numpy(cls, data, dtype=Float32):
        return data.astype(cls.dtypes[dtype])

    @classmethod
    def build_tensor(cls, data) -> np.ndarray:
        return np.asarray(data)
