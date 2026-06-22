from abc import abstractmethod

from qewton.backends.base import Backend, TensorType


class LinAlgBackend(Backend[TensorType]):
    """A Backend that implements linear algebra stuff such as
    matrix system solving, eigendecomposition, etc.
    """

    @staticmethod
    @abstractmethod
    def svd(x: TensorType) -> tuple[TensorType, TensorType, TensorType]:
        pass

    @staticmethod
    @abstractmethod
    def norm(
        x: TensorType, order: str | int = "fro", axis=None, keepdims=False
    ) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def det(x: TensorType) -> TensorType:
        pass
