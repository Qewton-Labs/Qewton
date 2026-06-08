from abc import abstractmethod
from typing import Callable

from qewton.backends.base import Backend, TensorType


class MathBackend(Backend[TensorType]):
    @classmethod
    @abstractmethod
    def add(cls, a, b):
        pass

    @classmethod
    @abstractmethod
    def mul(cls, a, b):
        pass
