from abc import abstractmethod
from typing import Any

from qewton.backends.base import Backend, TensorType


class LinAlgBackend(Backend[TensorType]):
    """A Backend that implements linear algebra stuff such as
    matrix system solving, eigendecomposition, etc.
    """

    @staticmethod
    @abstractmethod
    def svd(x: Any) -> tuple[TensorType, TensorType, TensorType]:
        pass
