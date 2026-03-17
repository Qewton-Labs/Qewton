class Backend:
    prefix = None


class TorchBackend(Backend):
    prefix = "torch"

    @classmethod
    def exists(cls):
        try:
            import torch

            return True
        except ImportError:
            return False

    @classmethod
    def standard_datatype(cls):
        if not TorchBackend.exists():
            raise ImportError("PyTorch is not installed.")
        import torch

        return torch.Tensor


class TensorflowBackend(Backend):
    prefix = "tensorflow"

    @classmethod
    def exists(cls):
        try:
            import tensorflow

            return True
        except ImportError:
            return False

    @classmethod
    def standard_datatype(cls):
        if not TensorflowBackend.exists():
            raise ImportError("TensorFlow is not installed.")
        import tensorflow as tf

        return tf.Tensor


class NumpyBackend(Backend):
    prefix = "numpy"

    @classmethod
    def exists(cls):
        try:
            import numpy

            return True
        except ImportError:
            return False

    @classmethod
    def standard_datatype(cls):
        if not NumpyBackend.exists():
            raise ImportError("NumPy is not installed.")
        import numpy as np

        return np.ndarray


DL_BACKEND_HIERARCHY = [TorchBackend, TensorflowBackend]
DEFAULT_DL_BACKEND = None
for backend in DL_BACKEND_HIERARCHY:
    if backend.exists():
        DEFAULT_DL_BACKEND = backend
        break
