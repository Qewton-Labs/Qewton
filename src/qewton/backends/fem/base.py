from __future__ import annotations
from typing import ClassVar

from qewton.backends.base import Backend, TensorType, ComputingBackend
from qewton.backends.fem.form import FormBackend
from qewton.backends.fem.function_space import FunctionSpaceBackend
from qewton.backends.fem.solve import SolveBackend
from qewton.backends.fem.mesh import MeshBackend
from qewton.config import dtypes as qt_dtypes


class FEMBackend(Backend[TensorType]):
    """A backend that solves differential equations using the finite element method."""

    form: ClassVar[type[FormBackend]]
    function_space: ClassVar[type[FunctionSpaceBackend]]
    comp: ClassVar[type[ComputingBackend]]
    solve: ClassVar[type[SolveBackend]]
    mesh: ClassVar[type[MeshBackend]]

    @classmethod
    def from_numpy(cls, data, dtype=qt_dtypes.Float32):
        """Converts a numpy array to the standard datatype of this backend.

        Args:
            data (np.ndarray): The numpy array to convert.

        Returns:
            TensorType: The converted data.
        """
        return data

    @classmethod
    def to_numpy(cls, tensor: TensorType) -> "np.ndarray":
        """Converts the backend tensor to a numpy array.

        Args:
            tensor (TensorType): The tensor to convert.

        Returns:
            np.ndarray: The converted numpy array.
        """
        return tensor
