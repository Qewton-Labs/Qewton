from __future__ import annotations
from pathlib import Path
from typing import Generic, TypeVar, ClassVar, TYPE_CHECKING, Protocol, Any

from qewton.config import dtypes as qt_dtypes
from qewton.config.devices import Device, cpu
from qewton.config.saving.loading import Deserializer
from qewton.config.saving.saving import Serializable, Serializer
from qewton.config.saving.schema_keys import SavingKeys

if TYPE_CHECKING:
    from qewton.backends.param import ParameterBackend
    from qewton.backends.grad import GradBackend
    from qewton.backends.math import MathBackend
    from qewton.backends.optim import OptimBackend
    from qewton.backends.nn import NNBackend
    from qewton.backends.linalg import LinAlgBackend
    from qewton.backends.random import RandomBackend


class ArrayLike(Protocol):

    def reshape(self, *new_shape) -> "TensorType": ...  # type: ignore

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def __getitem__(self, key) -> Any: ...

    def __setitem__(self, key, value): ...

    def __sub__(self, other) -> "TensorType": ...  # type: ignore

    def __add__(self, other) -> "TensorType": ...  # type: ignore

    def __mul__(self, other) -> "TensorType": ...  # type: ignore

    def __rmul__(self, other) -> "TensorType": ...  # type: ignore

    def __truediv__(self, other) -> "TensorType": ...  # type: ignore

    def __pow__(self, other) -> "TensorType": ...  # type: ignore


TensorType = TypeVar("TensorType", bound=ArrayLike)


class Backend(Serializable, Generic[TensorType]):
    """A Container that allows the connection of Qewton to any other library,
    which then might perform computations.
    Pre-defined subclasses require the implementation of certain methods.

    All backends are expected to be stateless, and only contain class methods.
    """

    default_dtype: ClassVar[type[TensorType]]  # type: ignore

    @classmethod
    def save(cls, serializer: Serializer) -> None:
        node_config = {
            SavingKeys.KEY_TYPE: SavingKeys.KEY_SERIALIZABLE,
            SavingKeys.KEY_CLASS: cls.__name__,
            SavingKeys.KEY_MODULE: cls.__module__,
        }
        serializer.set_serialization_data(id(cls), node_config)

    @classmethod
    def construct_new_object(cls, serializer: Deserializer, data_config: dict) -> Any:
        return cls  # Return the class itself, as backends are stateless and don't require instantiation


class ComputingBackend(Backend[TensorType]):
    math: ClassVar[type[MathBackend]]
    random: ClassVar[type[RandomBackend]]
    linalg: ClassVar[type[LinAlgBackend]]

    dtypes: ClassVar[dict]

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
    def build_tensor(
        cls, data, dtype: Any = qt_dtypes.Float32, device: Device | str = cpu
    ) -> TensorType:
        """Builds a tensor from the given data.

        Args:
            data: The data to build the tensor from.

        Returns:
            TensorType: The built tensor.
        """
        raise NotImplementedError(
            "The build_tensor method must be implemented by subclasses of Backend."
        )

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
    def cast_dtype(cls, data, dtype):
        """Changes the data type of the given data.

        Args:
            data (TensorType): The data to change the type of.
            dtype (qt_dtypes): The new type of the data.

        Returns:
            TensorType: The data with the new type.
        """
        raise NotImplementedError("The type changing is backend dependent")

    @classmethod
    def save_data(cls, data, path: str | Path):
        """Saves the given data to the given path.

        Args:
            data (TensorType): The data to save.
            path (str | Path): The path to save the data to.
        """
        raise NotImplementedError(
            "The saving method must be implemented by subclasses of Backend."
        )

    @classmethod
    def load_data(cls, path: str | Path) -> TensorType:
        """Loads the data from the given path.

        Args:
            path (str | Path): The path to load the data from.
        Returns:
            TensorType: The loaded data.
        """
        raise NotImplementedError(
            "The loading method must be implemented by subclasses of Backend."
        )


class DeepLearningBackend(ComputingBackend[TensorType]):
    """A Backend that implements all the necessary methods for deep learning.

    Note that this structure is similar to Keras' backends.
    """

    grad: ClassVar[type[GradBackend]]
    optim: ClassVar[type[OptimBackend]]
    nn: ClassVar[type[NNBackend]]
    param: ClassVar[type[ParameterBackend]]

    dtypes = {
        qt_dtypes.BFloat16: None,
        qt_dtypes.Float16: None,
        qt_dtypes.Float32: None,
        qt_dtypes.Float64: None,
        qt_dtypes.Complex32: None,
        qt_dtypes.Complex64: None,
        qt_dtypes.Complex128: None,
        qt_dtypes.UInt8: None,
        qt_dtypes.UInt16: None,
        qt_dtypes.UInt32: None,
        qt_dtypes.UInt64: None,
        qt_dtypes.Int8: None,
        qt_dtypes.Int16: None,
        qt_dtypes.Int32: None,
        qt_dtypes.Int64: None,
        qt_dtypes.Number: None,
        qt_dtypes.Bool: None,
    }

    @classmethod
    def from_numpy(cls, data, dtype=qt_dtypes.Float32):
        """Converts a numpy array to the standard datatype of this backend.

        Args:
            data (np.ndarray): The numpy array to convert.

        Returns:
            TensorType: The converted data.
        """
        raise NotImplementedError(
            "The from_numpy method must be implemented by subclasses of Backend."
        )

    @classmethod
    def get_device(cls, device: Device):
        raise NotImplementedError(
            "The get_device method must be implemented by subclasses of Backend."
        )
