import numpy as np

from qewton.backends.linalg import LinAlgBackend


class NumpyLinAlgBackend(LinAlgBackend[np.ndarray]):

    @staticmethod
    def svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.linalg.svd(x)

    @staticmethod
    def norm(x: np.ndarray, ord="fro", axis=None, keepdims=False) -> np.ndarray:
        return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
