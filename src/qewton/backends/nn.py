from abc import abstractmethod
from typing import Any, Literal

from qewton.backends.base import Backend, TensorType


class NNBackend(Backend[TensorType]):
    """A Backend that implements all neural network related operations, such as
    activations, layers, etc.
    """

    # region: activation functions

    @staticmethod
    @abstractmethod
    def relu(x: Any, /) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def sigmoid(x: Any, /) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def tanh(x: Any, /) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def softmax(x: Any, /, dim: int = -1) -> TensorType:
        pass

    # endregion

    # region: convolutional properties
    @staticmethod
    @abstractmethod
    def conv1d(
        x: Any,
        /,
        weight: Any,
        bias: Any | None = None,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def conv2d(
        x: Any,
        /,
        weight: Any,
        bias: Any | None = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def conv3d(
        x: Any,
        /,
        weight: Any,
        bias: Any | None = None,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def max_pool1d(
        x: Any,
        /,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] | None = None,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        ceil_mode: bool = False,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def max_pool2d(
        x: Any,
        /,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        ceil_mode: bool = False,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def max_pool3d(
        x: Any,
        /,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] | None = None,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        ceil_mode: bool = False,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def avg_pool1d(
        x: Any,
        /,
        kernel_size: int,
        stride: int = 0,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def avg_pool2d(
        x: Any,
        /,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def avg_pool3d(
        x: Any,
        /,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] | None = None,
        padding: int | tuple[int, int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def batch_norm1d(
        x: Any,
        running_mean: Any | None,
        running_var: Any | None,
        weight: Any | None = None,
        bias: Any | None = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def batch_norm2d(
        x: Any,
        running_mean: Any | None,
        running_var: Any | None,
        weight: Any | None = None,
        bias: Any | None = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def batch_norm3d(
        x: Any,
        running_mean: Any | None,
        running_var: Any | None,
        weight: Any | None = None,
        bias: Any | None = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def interpolate(
        x: Any,
        size: int | tuple[int] | tuple[int, int] | tuple[int, int, int] | None = None,
        scale_factor: float | tuple[float] | None = None,
        mode: Literal[
            "nearest", "linear", "bilinear", "bicubic", "trilinear"
        ] = "nearest",
        align_corners: bool | None = False,
    ) -> TensorType:
        pass
