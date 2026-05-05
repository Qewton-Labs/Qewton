from __future__ import annotations
from typing import Any, Generic, TypeVar

TensorType = TypeVar("TensorType")


class Backend(Generic[TensorType]):
    library: Any = None

    @classmethod
    def import_library(cls):
        """Imports the required backend library and returns it.

        Returns:
            module: the imported backend library.
        """
        raise NotImplementedError(
            "The import_ method must be implemented by subclasses of Backend."
        )

    @classmethod
    def exists(cls):
        try:
            cls.import_library()
            return True
        except ImportError:
            return False

    @classmethod
    def standard_datatype(cls):
        """Returns the standard datatype used by this backend (e.g. torch.Tensor for
        PyTorch).

        Returns:
            type: the standard datatype used by this backend.
        """
        raise NotImplementedError(
            "The standard_datatype method must be implemented by subclasses of Backend."
        )


def get_dtype_torch():
    try:
        import torch

        return torch.Tensor
    except ImportError:
        return Any


class TorchBackend(Backend[get_dtype_torch()]):
    @classmethod
    def import_library(cls):
        if cls.library is None:
            import torch  # pylint: disable=import-outside-toplevel # type: ignore

            cls.library = torch
        return cls.library

    @classmethod
    def standard_datatype(cls):
        if not cls.exists():
            raise ImportError("PyTorch is not installed.")

        return cls.library.Tensor


def get_dtype_tf():
    try:
        import tensorflow as tf

        return tf.Tensor
    except ImportError:
        return Any


class TensorflowBackend(Backend[get_dtype_tf()]):
    @classmethod
    def import_library(cls):
        if cls.library is None:
            import tensorflow as tf  # pylint: disable=import-outside-toplevel # type: ignore

            cls.library = tf
        return cls.library

    @classmethod
    def standard_datatype(cls):
        if not cls.exists():
            raise ImportError("TensorFlow is not installed.")

        return cls.library.Tensor


DL_BACKEND_HIERARCHY = [TorchBackend, TensorflowBackend]
# TODO: could be slow due to unnecessary imports
DEFAULT_DL_BACKEND = None
for backend in DL_BACKEND_HIERARCHY:
    if backend.exists():
        DEFAULT_DL_BACKEND = backend
        break
