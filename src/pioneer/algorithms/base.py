import warnings
from abc import abstractmethod
from enum import Enum, auto


from ..config.configuration_base import DataConfiguration
from ..config.variables import Variable
from ..config.axis import FeatureAxis
from ..nodes.base import Node, Port


class AlgorithmState(Enum):
    # TODO: Is this needed?
    FIXED = 1
    UNINITIALIZED = 2
    READY = 3
    TRAINED = 4


class AlgorithmAttributes(Enum):
    SYMMETRIC = auto()  # if a "flipped" input yields the same output
    TRANSLATION_INVARIANT = auto()
    ROTATION_INVARIANT = auto()
    LINEAR = auto()
    DIFFERENTIABLE = auto()  # the output is differentiable in regards to the input
    INVERTIBLE = auto()
    NORMALIZES_DATA = auto()
    DETERMINISTIC = auto()  # the run call (diffusion models for example not)
    TRAINABLE = auto()  # TODO:Is this needed?
    OUTPUTS_PROBABILITIES = auto()  # useful for classifiers?
    GPU_ACCELERATED = auto()
    MUTATES_INPUT = auto()  # if input is changed in-place
    SUPPORTS_MISSING_VALUES = auto()  # if values like NaN are handled


class AlgorithmNode(Node):
    """General node representing an algorithm that should solve a given problem
    or a part of it.
    """

    attributes: set[AlgorithmAttributes] = set()  # base class static-default

    def __init__(
        self,
        input_variable: Variable,
        output_variable: Variable,
        name: str = "AlgorithmNode",
    ) -> None:
        """
        Args:
            input_variable (Variable): The input variables of the algorithm.
            output_variable (Variable): The output variables of the algorithm.
            name (str, optional): The name of the node. Defaults to "AlgorithmNode".
        """
        super().__init__(name=name)
        self.input_variable = input_variable
        self.output_variable = output_variable
        self._state: AlgorithmState = AlgorithmState.UNINITIALIZED

    def fulfills(self, constraint, data=None) -> bool:
        # return True or an empirical measure on how well a constraint is
        # fulfilled (if data is available)?
        raise NotImplementedError("Fulfills method not implemented.")

    @abstractmethod
    def setup(self) -> None:
        """Creates the underlying algorithm instance (e.g. creates the
        neural network)

        This should not happen in the __init__ call, given that in the
        HyperParameter tuning we need to recreated the underlying algorithm
        instance, but dont want to create a new node inside our graph.
        """

    @property
    def state(self) -> AlgorithmState:
        return self._state

    def fix_algorithm_state(self) -> None:
        """Fix all properties of the algorithm so it will not be
        trained or recreated!
        """
        if self.state == AlgorithmState.UNINITIALIZED:
            warnings.warn(
                "This Algorithm is not initialized, fixing it now may lead \
                    to unexpected behavior. Maybe call .setup() first?",
                UserWarning,
            )
            return
        self._state = AlgorithmState.FIXED

    @property
    def input_ports(self) -> dict[str, Port]:
        data_axis = FeatureAxis(variables=self.input_variable)
        data_config = DataConfiguration(None, [..., data_axis], feature_axis=data_axis)
        return {self.InputKeys.INPUT: Port(data_config, self, "in_port", True)}

    @property
    def output_ports(self) -> dict[str, Port]:
        data_axis = FeatureAxis(variables=self.output_variable)
        data_config = DataConfiguration(None, [..., data_axis], feature_axis=data_axis)
        return {self.OutputKeys.OUTPUT: Port(data_config, self, "out_port")}
