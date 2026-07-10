import torch

from qewton.backends.torch.device import get_torch_device
from qewton.config.devices import Device, cpu
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
from qewton.backends.torch.nn import TorchNNBackend
from qewton.backends.torch.optim import TorchOptimBackend
from qewton.backends.torch.linalg import TorchLinAlgBackend
from qewton.backends.torch.param import TorchParameterBackend
from qewton.backends.torch.math import TorchMathBackend
from qewton.backends.torch.random import TorchRandomBackend


class TorchBackend(DeepLearningBackend[torch.Tensor]):
    """
    The Backend that implements all functionalities in PyTorch.
    """

    math = TorchMathBackend
    nn = TorchNNBackend
    grad = TorchGradBackend
    optim = TorchOptimBackend
    linalg = TorchLinAlgBackend
    param = TorchParameterBackend
    random = TorchRandomBackend

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
    def to(cls, data: torch.Tensor, device):
        return data.to(cls.get_device(device))

    @classmethod
    def load(cls, path, **kwargs):
        return torch.load(path, **kwargs)

    @classmethod
    def from_numpy(cls, data, dtype=Float32):
        converted_type = cls.dtypes.get(dtype, dtype)
        return torch.from_numpy(data).to(converted_type)

    @classmethod
    def get_device(cls, device: Device):
        return get_torch_device(device)

    @classmethod
    def build_tensor(
        cls, data, dtype=Float32, device: Device | str = cpu
    ) -> torch.Tensor:
        converted_type = cls.dtypes.get(dtype, dtype)
        if isinstance(data, torch.Tensor):
            return data.to(dtype=converted_type)
        return torch.as_tensor(data, dtype=converted_type, device=cls.get_device(device))

    @classmethod
    def cast_dtype(cls, data: torch.Tensor, dtype):
        return data.type(cls.dtypes.get(dtype, dtype))
