import warnings
from abc import abstractmethod
from enum import Enum


from ..configurations.configuration_base import DataConfiguration
from ..configurations.variables import Variable
from ..configurations.axis import FeatureAxis
from ..nodes.base import Node, Port


class AlgorithmState(Enum):
    FIXED = 1
    UNINITIALIZED = 2
    READY = 3
    TRAINED = 4


class AlgorithmNode(Node):

    def __init__(
        self,
        input_variable: Variable,
        output_variable: Variable,
        name: str = "AlgorithmNode",
    ) -> None:
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
