from qewton.backends.base import Backend, TensorType


class LinalgBackend(Backend[TensorType]):
    """A Backend that implements linear algebra stuff such as
    matrix system solving, eigendecomposition, etc.
    """
