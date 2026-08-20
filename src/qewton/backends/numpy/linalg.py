import numpy as np

from qewton.backends.linalg import LinAlgBackend


class NumpyLinAlgBackend(LinAlgBackend[np.ndarray]):

    @staticmethod
    def svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, S, V = np.linalg.svd(x)
        return U, S, V.T

    @staticmethod
    def pca(x: np.ndarray, q=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, S, V = np.linalg.svd(x)
        return U[:, :q], S[:q], V[:, :q]

    @staticmethod
    def norm(x: np.ndarray, order="fro", axis=None, keepdims=False) -> np.ndarray:
        return np.linalg.norm(x, ord=order, axis=axis, keepdims=keepdims)

    @staticmethod
    def det(x: np.ndarray) -> np.ndarray:
        return np.linalg.det(x)

    @staticmethod
    def inv(x: np.ndarray) -> np.ndarray:
        return np.linalg.inv(x)
