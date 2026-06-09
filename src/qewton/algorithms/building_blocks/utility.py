from typing import Annotated

from qewton.algorithms.backend_node import BackendNode, TensorType
from qewton.config.backend import (
    DEFAULT_DL_BACKEND,
    Backend,
)
from qewton.config.axes import EllipsisAxes
from qewton.config.data_configurations import DataConfiguration


class Cast(BackendNode[TensorType]):
    ellipsis_axes = EllipsisAxes()

    def __init__(
        self,
        dtype_name: str,
        name: str = "complex_valued",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(name, backend)
        self.dtype = {
            "bfloat16": backend.library.bfloat16,
            "float16": backend.library.float16,
            "half": backend.library.float16,
            "float32": backend.library.float32,
            "float": backend.library.float32,
            "float64": backend.library.float64,
            "double": backend.library.float64,
            "complex32": backend.library.complex32,
            "chalf": backend.library.complex32,
            "complex64": backend.library.complex64,
            "cfloat": backend.library.complex64,
            "complex128": backend.library.complex128,
            "cdouble": backend.library.complex128,
            "uint8": backend.library.uint8,
            "uint16": backend.library.uint16,
            "uint32": backend.library.uint32,
            "uint64": backend.library.uint64,
            "int8": backend.library.int8,
            "int16": backend.library.int16,
            "int32": backend.library.int32,
            "int64": backend.library.int64,
            "bool": backend.library.bool,
        }[dtype_name]

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Annotated[TensorType, DataConfiguration(ellipsis_axes)]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return x.type(self.dtype)

    def tensorflow_implementation(self, x):
        return self.backend.cast(x, self.dtype)


class GetShape(BackendNode[TensorType]):
    def __init__(
        self,
        name: str = "get_shape",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(name, backend)

    def forward(
        self, x: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        return tuple(x.shape)


class Reshape(BackendNode[TensorType]):
    def __init__(
        self,
        name: str = "view",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(name, backend)

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration.empty()],
        shape: Annotated[TensorType, DataConfiguration.empty()],
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        return self.backend.library.reshape(x, shape)
