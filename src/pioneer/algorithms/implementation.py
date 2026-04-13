from typing import Iterable


class Implementation:
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

    @classmethod
    def create_trainable_parameter(
        cls,
        param_shape: tuple[int, ...],
    ):
        """Returns the a trainable parameter object in this type of
        implementation (e.g. torch.nn.Parameter for PyTorch).

        Args:
            param_shape (tuple[int, ...]): The desired shape of the parameter.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(
            "The creation of parameters is implemented in the corresponding subclass"
        )

    @property
    # TODO: Is this still needed?
    def trainable_parameters(self) -> Iterable:
        """Returns the trainable parameters of this node, which can be used for
        training the underlying algorithm (e.g. a neural network).

        Returns:
            Iterable of the trainable parameters
        """
        raise NotImplementedError(
            "The trainable_parameters property must be implemented by subclasses \
                of Implementation."
        )

    def to(self, device):
        """Moves the underlying algorithm to the specified device (e.g. GPU).

        Args:
            device: the device to move the underlying algorithm to.
        """
        # By default, do nothing. Subclasses can override this if needed.


class TorchImplementation(Implementation):
    """
    A PyTorch implementation consisting of a single torch.nn.Module
    """

    def __init__(self, torch_module) -> None:
        """Creates the underlying PyTorch module instance."""
        super().__init__()
        self._torch_module = torch_module

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

    @property
    def torch_module(self):
        return self._torch_module

    def __call__(self, *args, **kwargs):
        return self._torch_module(*args, **kwargs)

    @property
    def trainable_parameters(self) -> Iterable:
        """Returns the trainable parameters of this node, which can be used for
        training the underlying algorithm (e.g. a neural network).

        Returns:
            _TrainableParameterBase: trainable parameters
        """
        return self._torch_module.parameters()

    @classmethod
    def create_trainable_parameter(
        cls,
        param_shape: tuple[int, ...],
    ):
        import torch  # type: ignore

        # TODO: Should be more generalized, just some first version to try this
        # out. Grad maybe also as an input, and how handle the initial choice?
        param = torch.nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if len(param_shape) > 1:
            torch.nn.init.xavier_uniform_(param)
        return param

    def to(self, device):
        """Moves the underlying algorithm to the specified device (e.g. GPU).

        Args:
            device: the device to move the underlying algorithm to.
        """
        self._torch_module.to(device)


class TensorflowImplementation(Implementation):
    """
    A TensorFlow implementation consisting of a single tf.keras.Layer
    """

    def __init__(self, tf_layer) -> None:
        """Creates the underlying TensorFlow layer instance."""
        super().__init__()
        self._tf_layer = tf_layer

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

    @property
    def tf_layer(self):
        return self._tf_layer

    def __call__(self, *args, **kwargs):
        return self._tf_layer(*args, **kwargs)

    @property
    def trainable_parameters(self) -> Iterable:
        """Returns the trainable parameters of this node, which can be used for
        training the underlying algorithm (e.g. a neural network).

        Returns:
            _TrainableParameterBase: trainable parameters
        """
        return self._tf_layer.trainable_variables  # TODO


DL_IMPLEMENTATION_HIERARCHY = [TorchImplementation, TensorflowImplementation]
# TODO: could be slow due to unnecessary imports
DEFAULT_DL_IMPLEMENTATION = None
for implementation in DL_IMPLEMENTATION_HIERARCHY:
    if implementation.exists():
        DEFAULT_DL_IMPLEMENTATION = implementation
        break
