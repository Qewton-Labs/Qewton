from typing import Any

from qewton.config.backend import (
    Backend,
    TorchBackend,
    TensorflowBackend,
    DEFAULT_DL_BACKEND,
)


class Optimizer:
    """
    Base class for all optimizers.

    This class provides a common interface for building optimizer instances
    for different deep learning backends (e.g., PyTorch, TensorFlow).
    """

    existing_implementations = {}
    requires_closure = False

    def __init__(self, backend: Backend = DEFAULT_DL_BACKEND) -> None:
        """
        Initializes the Optimizer with a specified backend.
        Args:
            backend (Backend, optional): The deep learning backend to use.
                                         Defaults to DEFAULT_DL_BACKEND.
        """
        self.backend = backend
        _ = self.backend.import_library()

    def build_optimizer(self):
        """
        Abstract method to build and return an optimizer instance specific to the backend.
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Optimizers are implemented via subclasses.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Allows the optimizer instance to be called directly, building and returning
        the backend-specific optimizer with the provided arguments.
        """
        optimizer_obj = self.build_optimizer()
        return optimizer_obj(*args, **kwds)


##################################################################################
# TODO: Add more optimizers and their respective implementations here.
# Also add some documentation.


# TODO: Maybe we need some ordering of the inputs again between different backends?
class Adam(Optimizer):
    """
    Adam optimizer implementation for various backends.
    """

    def build_optimizer(self):
        """
        Builds and returns the Adam optimizer from the specified backend library.
        Returns:
            Any: The backend-specific Adam optimizer class or function.
        Raises:
            NotImplementedError: If Adam is not implemented for the current backend.
        """
        if self.backend == TorchBackend:
            return self.backend.library.optim.Adam
        if self.backend == TensorflowBackend:
            return self.backend.library.keras.optimizers.Adam
        raise NotImplementedError(
            f"No implementation of Adam exists for backend {self.backend}."
        )


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer implementation for various backends.
    """

    def build_optimizer(self):
        """
        Builds and returns the SGD optimizer from the specified backend library.
        Returns:
            Any: The backend-specific SGD optimizer class or function.
        Raises:
            NotImplementedError: If SGD is not implemented for the current backend.
        """
        if self.backend == TorchBackend:
            return self.backend.library.optim.SGD
        if self.backend == TensorflowBackend:
            return self.backend.library.keras.optimizers.SGD
        raise NotImplementedError(
            f"No implementation of SGD exists for backend {self.backend}."
        )


class LBFGS(Optimizer):
    """
    L-BFGS optimizer implementation.
    """

    def build_optimizer(self):
        """
        Builds and returns the L-BFGS optimizer from the specified backend library.
        Returns:
            Any: The backend-specific L-BFGS optimizer class or function.
        Raises:
            NotImplementedError: If L-BFGS is not implemented for the current backend.
        """
        if self.backend == TorchBackend:
            return self.backend.library.optim.LBFGS
        raise NotImplementedError(
            f"No implementation of LBFGS exists for backend {self.backend}."
        )

    requires_closure = True
