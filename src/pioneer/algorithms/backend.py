from typing import Any


class Backend:
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


class TorchBackend(Backend):
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


class TensorflowBackend(Backend):
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
