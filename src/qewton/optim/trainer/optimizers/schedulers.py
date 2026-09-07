from typing import Any

from qewton.backends import DEFAULT_DL_BACKEND
from qewton.backends.base import DeepLearningBackend
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class LR_Scheduler:
    """
    Base class for all learning rate schedulers.
    """

    requires_closure = False

    def __init__(self, backend: type[DeepLearningBackend] = DEFAULT_DL_BACKEND) -> None:
        """
        Initializes the learning rate scheduler with a specified backend.
        Args:
            backend (Backend, optional): The deep learning backend to use.
                                         Defaults to DEFAULT_DL_BACKEND.
        """
        self.backend = backend

    def build_scheduler(self, optimizer_obj: Any) -> Any:
        """
        Abstract method to build and return an scheduler instance specific to the backend.
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Optimizers are implemented via subclasses.")

    def hyper_parameters(self) -> set[HyperParameter]:
        """
        Returns the set of HyperParameter instances referenced by this scheduler.
        Returns:
            set[HyperParameter]: Hyperparameters used by this scheduler.
        """
        hp_set = set[HyperParameter]()
        for attr_name in dir(self):
            if isinstance(getattr(self, attr_name), HyperParameter):
                hp_set.add(getattr(self, attr_name))
        return hp_set


##################################################################################
class StepLR(LR_Scheduler):
    """
    A StepLR scheduler implementation for various backends.
    """

    def __init__(
        self,
        step_size: int | HyperParameter,
        gamma: float | HyperParameter,
        backend: type[DeepLearningBackend] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(backend)
        self.step_size = HyperParameter.from_value(step_size, "Step Size")
        self.gamma = HyperParameter.from_value(gamma, "Gamma")

    def build_scheduler(self, optimizer_obj: Any) -> Any:
        """
        Builds and returns the StepLR scheduler from the specified backend library.
        Returns:
            Any: The backend-specific StepLR scheduler class or function.
        Raises:
            NotImplementedError: If StepLR is not implemented for the current backend.
        """
        return self.backend.optim.step_lr(
            optimizer_obj, self.step_size.value, self.gamma.value
        )


class ExponentialLR(LR_Scheduler):
    """
    An ExponentialLR scheduler implementation for various backends.
    """

    def __init__(
        self,
        gamma: float | HyperParameter,
        backend: type[DeepLearningBackend] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(backend)
        self.gamma = HyperParameter.from_value(gamma, "Gamma")

    def build_scheduler(self, optimizer_obj: Any) -> Any:
        """
        Builds and returns the ExponentialLR scheduler from the specified backend library.
        Returns:
            Any: The backend-specific ExponentialLR scheduler class or function.
        Raises:
            NotImplementedError: If ExponentialLR is not implemented for the current backend.
        """
        return self.backend.optim.exponential_lr(optimizer_obj, self.gamma.value)


class CosineAnnealingLR(LR_Scheduler):
    """
    A CosineAnnealingLR scheduler implementation for various backends.
    """

    def __init__(
        self,
        T_max: int | HyperParameter,
        eta_min: float | HyperParameter = 0.0,
        backend: type[DeepLearningBackend] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(backend)
        self.T_max = HyperParameter.from_value(T_max, "T_max")
        self.eta_min = HyperParameter.from_value(eta_min, "Eta Min")

    def build_scheduler(self, optimizer_obj: Any) -> Any:
        """
        Builds and returns the CosineAnnealingLR scheduler from the specified backend library.
        Returns:
            Any: The backend-specific CosineAnnealingLR scheduler class or function.
        Raises:
            NotImplementedError: If CosineAnnealingLR is not implemented for the current backend.
        """
        return self.backend.optim.cosine_annealing_lr(
            optimizer_obj, self.T_max.value, self.eta_min.value
        )
