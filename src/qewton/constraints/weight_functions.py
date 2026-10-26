import math

from qewton.optim.parameters.hyperparameter_base import HyperParameter


class ConstraintWeightFunction:
    """
    Base class for constraint weight functions.
    """

    def __call__(self, iteration):
        """
        Returns the weight for the given constraint at the given iteration.

        Args:
            iteration: The current iteration number.

        Returns:
            The weight for the given constraint at the given iteration.
        """
        raise NotImplementedError("ConstraintWeightFunction is an abstract base class.")

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        """
        Returns a list of hyperparameters for this weight function.

        Returns:
            A list of hyperparameters for this weight function.
        """
        return [v for v in vars(self).values() if isinstance(v, HyperParameter)]


class LinearRampWeight(ConstraintWeightFunction):
    """Linearly ramps the constraint weight from ``initial_weight`` to
    ``target_weight`` between ``start_iteration`` and ``full_iteration``.

    For ``iteration <= start_iteration`` the returned weight is
    ``initial_weight``. For ``iteration >= full_iteration`` the returned
    weight is ``target_weight``.

    Args:
        start_iteration (int | HyperParameter): The iteration at which the
            ramp starts.
        full_iteration (int | HyperParameter): The iteration at which the
            ramp ends and the target weight is reached.
        initial_weight (float | HyperParameter, optional): The weight at the
            start of the ramp. Defaults to 0.0.
        target_weight (float | HyperParameter, optional): The weight at the
            end of the ramp. Defaults to 1.0.
    """

    def __init__(
        self,
        start_iteration: int | HyperParameter,
        full_iteration: int | HyperParameter,
        initial_weight: float | HyperParameter = 0.0,
        target_weight: float | HyperParameter = 1.0,
    ):
        self.start_iteration = HyperParameter.from_value(
            start_iteration, "Ramp Start Iteration"
        )
        self.full_iteration = HyperParameter.from_value(
            full_iteration, "Ramp Full Iteration"
        )
        self.initial_weight = HyperParameter.from_value(initial_weight, "Initial Weight")
        self.target_weight = HyperParameter.from_value(target_weight, "Target Weight")

    def __call__(self, iteration):
        start = int(self.start_iteration.value)
        full = int(self.full_iteration.value)
        w0 = float(self.initial_weight.value)
        w1 = float(self.target_weight.value)

        if full <= start:
            return w1 if iteration >= start else w0
        if iteration <= start:
            return w0
        if iteration >= full:
            return w1

        alpha = (iteration - start) / (full - start)
        return w0 + alpha * (w1 - w0)


class StepWeight(ConstraintWeightFunction):
    """Keeps a low weight until ``activate_iteration`` and then jumps up.

    This is useful when a constraint should only become active after some
    warm-up iterations.

    Args:
        activate_iteration (int | HyperParameter): The iteration at which the
            weight jumps up.
        inactive_weight (float | HyperParameter, optional): The weight before
            the activation iteration. Defaults to 0.0.
        active_weight (float | HyperParameter, optional): The weight after the
            activation iteration. Defaults to 1.0.
    """

    def __init__(
        self,
        activate_iteration: int | HyperParameter,
        inactive_weight: float | HyperParameter = 0.0,
        active_weight: float | HyperParameter = 1.0,
    ):
        self.activate_iteration = HyperParameter.from_value(
            activate_iteration, "Activate Iteration"
        )
        self.inactive_weight = HyperParameter.from_value(
            inactive_weight, "Inactive Weight"
        )
        self.active_weight = HyperParameter.from_value(active_weight, "Active Weight")

    def __call__(self, iteration):
        threshold = int(self.activate_iteration.value)
        if iteration < threshold:
            return float(self.inactive_weight.value)
        return float(self.active_weight.value)


class CosineRampWeight(ConstraintWeightFunction):
    """Smoothly ramps weight from ``initial_weight`` to ``target_weight``.

    Uses a half-cosine schedule between ``start_iteration`` and
    ``full_iteration`` to avoid abrupt optimization changes.

    Args:
        start_iteration (int | HyperParameter): The iteration at which the
            ramp starts.
        full_iteration (int | HyperParameter): The iteration at which the
            ramp ends and the target weight is reached.
        initial_weight (float | HyperParameter, optional): The weight at the
            start of the ramp. Defaults to 0.0.
        target_weight (float | HyperParameter, optional): The weight at the
            end of the ramp. Defaults to 1.0.
    """

    def __init__(
        self,
        start_iteration: int | HyperParameter,
        full_iteration: int | HyperParameter,
        initial_weight: float | HyperParameter = 0.0,
        target_weight: float | HyperParameter = 1.0,
    ):
        self.start_iteration = HyperParameter.from_value(
            start_iteration, "Cosine Start Iteration"
        )
        self.full_iteration = HyperParameter.from_value(
            full_iteration, "Cosine Full Iteration"
        )
        self.initial_weight = HyperParameter.from_value(initial_weight, "Initial Weight")
        self.target_weight = HyperParameter.from_value(target_weight, "Target Weight")

    def __call__(self, iteration):
        start = int(self.start_iteration.value)
        full = int(self.full_iteration.value)
        w0 = float(self.initial_weight.value)
        w1 = float(self.target_weight.value)

        if full <= start:
            return w1 if iteration >= start else w0
        if iteration <= start:
            return w0
        if iteration >= full:
            return w1

        alpha = (iteration - start) / (full - start)
        smooth_alpha = 0.5 * (1.0 - math.cos(math.pi * alpha))
        return w0 + smooth_alpha * (w1 - w0)
