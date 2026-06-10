from __future__ import annotations
from typing import Generic, TypeVar, ClassVar

from qewton.config import dtypes as qt_dtypes
from qewton.backends.grad import GradBackend
from qewton.backends.math import MathBackend
from qewton.backends.optim import OptimBackend
from qewton.backends.nn import NNBackend

TensorType = TypeVar("TensorType")


# TODO: Wa have standard datatype gives tensors, but what about float, etc.?


class Backend(Generic[TensorType]):
    """A Container that allows the connection of Qewton to any other library,
    which then might perform computations.
    Pre-defined subclasses require the implementation of certain methods.

    All backends are expected to be stateless, and only contain class methods.
    """


class DeepLearningBackend(Backend[TensorType]):
    """A Backend that implements all the necessary methods for deep learning.

    Note that this structure is similar to Keras' backends.
    """

    math: ClassVar[type[MathBackend]]
    grad: ClassVar[type[GradBackend]]
    optim: ClassVar[type[OptimBackend]]
    nn: ClassVar[type[NNBackend]]

    default_dtype: ClassVar[type[TensorType]]  # type: ignore

    dtypes: ClassVar[dict]  # type: ignore

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # check only non abstract classes
        required_keys = [
            qt_dtypes.BFloat16,
            qt_dtypes.Float16,
            qt_dtypes.Float32,
            qt_dtypes.Float64,
            qt_dtypes.Complex32,
            qt_dtypes.Complex64,
            qt_dtypes.Complex128,
            qt_dtypes.UInt8,
            qt_dtypes.UInt16,
            qt_dtypes.UInt32,
            qt_dtypes.UInt64,
            qt_dtypes.Int8,
            qt_dtypes.Int16,
            qt_dtypes.Int32,
            qt_dtypes.Int64,
            qt_dtypes.Number,
            qt_dtypes.Bool,
        ]

        if not hasattr(cls, "dtypes") or not isinstance(cls.dtypes, dict):
            raise TypeError(f"Backend {cls.__name__} requires a 'dtypes' dictionary.")

        missing = [str(k) for k in required_keys if k not in cls.dtypes]
        if missing:
            raise KeyError(f"In {cls.__name__}.dtypes the following keys are missing:\
                    {', '.join(missing)}")

    @classmethod
    def to(cls, data, device):
        """Moves the data to the given device.

        Args:
            data (TensorType): The data to move.
            device (str): The device to move the data to.

        Returns:
            TensorType: The moved data.
        """
        raise NotImplementedError(
            "The moving to a different device is backend dependent."
        )

    @classmethod
    def from_numpy(cls, data):
        """Converts a numpy array to the standard datatype of this backend.

        Args:
            data (np.ndarray): The numpy array to convert.

        Returns:
            TensorType: The converted data.
        """
        raise NotImplementedError(
            "The from_numpy method must be implemented by subclasses of Backend."
        )
