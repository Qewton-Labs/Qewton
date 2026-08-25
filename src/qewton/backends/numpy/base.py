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

from qewton.backends.base import ComputingBackend
from qewton.backends.numpy.math import NumpyMathBackend
from qewton.backends.numpy.random import NumpyRandomBackend
from qewton.backends.numpy.linalg import NumpyLinAlgBackend
from qewton.config.devices import Device, cpu


class NumPyBackend(ComputingBackend[np.ndarray]):
    default_dtype = np.ndarray

    math = NumpyMathBackend
    random = NumpyRandomBackend
    linalg = NumpyLinAlgBackend

    dtypes = {
        BFloat16: False,
        Float16: np.float16,
        Float32: np.float32,
        Float64: np.float64,
        Complex32: None,
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
        converted_type = cls.dtypes.get(dtype, dtype)
        return data.astype(converted_type)

    @classmethod
    def build_tensor(cls, data, dtype=Float32, device: Device | str = cpu) -> np.ndarray:
        converted_type = cls.dtypes.get(dtype, dtype)
        return np.asarray(data, dtype=converted_type)

    @classmethod
    def get_device(cls, device):
        return "cpu"  # NumPy always operates on CPU

    @classmethod
    def to(cls, data, device):
        return data  # always on cpu

    @classmethod
    def cast_dtype(cls, data: np.ndarray, dtype):
        return data.astype(cls.dtypes.get(dtype, dtype))

    @classmethod
    def to_numpy(cls, data: np.ndarray) -> np.ndarray:
        return data
