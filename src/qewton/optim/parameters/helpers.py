from enum import Enum


class HyperParameterState(Enum):
    """Denotes the state of a HyperParameter and how it is handled in the
    tuning process.

    FIXED
        The Hyperparameter does not change and is ignored while tuning.
    OPTIMIZE
        The Hyperparameter is changed in the tuning process.
    """

    FIXED = 1
    OPTIMIZE = 2


class HyperParameterScale(Enum):
    """Influence the sampling of HyperParameters.

    LINEAR
        A uniform sampling in the given parameter range.
    LOG
        A uniform logarithmic sampling in the given parameter range.
    POWER
        Sampling via a power law x**power, where power can be set in the
        HyperParameter class. power > 1 leads to a bias toward lower values,
        p < 1 leads to a bias toward higher values.
    """

    LINEAR = "linear"
    LOG = "log"
    POWER = "power"


class HyperParameterCondition:
    def __init__(self, func, deps: set):
        self.func = func
        self.deps = deps

    def evaluate(self, config) -> bool:
        if config is not None:
            for node in self.deps:
                if not node.name in config:
                    return False
        return self.func(config)

    def __and__(self, other):
        return _HyperParameterConditionCombination(
            lambda config: self.evaluate(config) and other.evaluate(config),
            self.deps.union(other.deps),
        )

    def __or__(self, other):
        return _HyperParameterConditionCombination(
            lambda config: self.evaluate(config) or other.evaluate(config),
            self.deps.union(other.deps),
        )

    def __invert__(self):
        return _HyperParameterConditionCombination(
            lambda config: not self.evaluate(config), deps=self.deps
        )

    def __bool__(self):
        raise RuntimeError(
            "Condition objects cannot be used with 'if' or 'not'. "
            "Use .evaluate() explicitly."
        )


class _HyperParameterConditionCombination(HyperParameterCondition):
    """Helper class to have a more efficient evaluate function."""

    def evaluate(self, config) -> bool:
        return self.func(config)
