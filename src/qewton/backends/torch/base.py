from typing import Annotated

import torch
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
from qewton.backends.torch.grad import TorchGradBackend
from qewton.backends.torch.math import TorchMathBackend
from qewton.backends.torch.nn import TorchNNBackend
from qewton.backends.torch.optim import TorchOptimBackend


class TorchBackend(DeepLearningBackend[torch.Tensor]):
    math = TorchMathBackend
    nn = TorchNNBackend
    grad = TorchGradBackend
    optim = TorchOptimBackend

    default_dtype = torch.Tensor

    dtypes = {
        BFloat16: torch.bfloat16,
        Float16: torch.float16,
        Float32: torch.float32,
        Float64: torch.float64,
        Complex32: torch.complex32,
        Complex64: torch.complex64,
        Complex128: torch.complex128,
        UInt8: torch.uint8,
        UInt16: torch.uint16,
        UInt32: torch.uint32,
        UInt64: torch.uint64,
        Int8: torch.int8,
        Int16: torch.int16,
        Int32: torch.int32,
        Int64: torch.int64,
        Number: None,
        Bool: torch.bool,
    }

    @classmethod
    def to(cls, data, device):
        return data.to(device)

    @classmethod
    def from_numpy(cls, data, dtype: Annotated = Float32):
        return torch.from_numpy(data).to(dtype=cls.dtypes[dtype])
