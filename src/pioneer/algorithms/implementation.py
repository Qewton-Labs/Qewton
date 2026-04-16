from typing import Iterable


class Implementation:

    def __init__(self, implementation_name, inputs=None, outputs=None):
        self.implementation_name = implementation_name
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "The __call__ method must be implemented by subclasses of Implementation."
        )

    @classmethod
    def exists(cls):
        """Checks if the required backend for this implementation exists.

        Returns:
            bool: True if the required backend exists, False otherwise.
        """
        raise NotImplementedError(
            "The exists method must be implemented by subclasses of Implementation."
        )

    @classmethod
    def standard_datatype(cls):
        """Returns the standard datatype used by this implementation (e.g. torch.Tensor
        for PyTorch).

        Returns:
            type: the standard datatype used by this implementation.
        """
        raise NotImplementedError(
            "The standard_datatype method must be implemented by subclasses of\
                Implementation."
        )


class TorchImplementation(Implementation):
    """
    A PyTorch implementation consisting of a single torch.nn.Module
    """

    def __init__(self, implementation_name, inputs=None, outputs=None) -> None:
        """Creates the underlying PyTorch module instance."""
        super().__init__(implementation_name, inputs, outputs)
        import torch  # pylint: disable=import-outside-toplevel # type: ignore

        self._torch_module = getattr(torch, self.implementation_name)

    @classmethod
    def exists(cls):
        try:
            import torch  # pylint: disable=unused-import # type: ignore

            return True
        except ImportError:
            return False

    @classmethod
    def standard_datatype(cls):
        if not cls.exists():
            raise ImportError("PyTorch is not installed.")
        import torch

        return torch.Tensor

    def __call__(self, *args, **kwargs):
        return self._torch_module(*args, **kwargs)


class TensorflowImplementation(Implementation):
    """
    A TensorFlow implementation consisting of a single tf layer
    """

    def __init__(self, implementation_name, inputs=None, outputs=None) -> None:
        """Creates the underlying TensorFlow layer instance."""
        super().__init__(implementation_name, inputs, outputs)
        import tensorflow as tf  # pylint: disable=import-error # type: ignore

        self._tf_layer = getattr(tf, self.implementation_name)

    @classmethod
    def exists(cls):
        try:
            import tensorflow  # pylint: disable=unused-import # type: ignore

            return True
        except ImportError:
            return False

    @classmethod
    def standard_datatype(cls):
        if not cls.exists():
            raise ImportError("TensorFlow is not installed.")
        import tensorflow as tf  # pylint: disable=import-error # type: ignore

        return tf.Tensor

    def __call__(self, *args, **kwargs):
        return self._tf_layer(*args, **kwargs)


DL_IMPLEMENTATION_HIERARCHY = [TorchImplementation, TensorflowImplementation]
# TODO: could be slow due to unnecessary imports
DEFAULT_DL_IMPLEMENTATION = None
for implementation in DL_IMPLEMENTATION_HIERARCHY:
    if implementation.exists():
        DEFAULT_DL_IMPLEMENTATION = implementation
        break
