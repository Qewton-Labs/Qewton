import warnings
from abc import abstractmethod
from enum import Enum


from ..data.configurations.configuration_base import DataConfiguration
from ..data.configurations.variables import Variable
from ..data.configurations.axis import FeatureAxis
from ..pipeline.nodes.base import Node, Port, InputPortDictionary, OutputPortDictionary


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

    def fulfills(self, constraint, data=None):
        # return True or an empirical measure on how well a constraint is
        # fulfilled (if data is available)?
        raise NotImplementedError("Fulfills method not implemented.")

    @abstractmethod
    def setup(self) -> None:
        """Creates the underlying algorithm instance (e.g. creates the
        neural network)
        """

    @property
    def state(self):
        return self._state

    def fix_algorithm_state(self):
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
    def input_ports(self) -> InputPortDictionary:
        data_axis = FeatureAxis(variables=self.input_variable)
        data_config = DataConfiguration(None, [..., data_axis], feature_axis=data_axis)
        return {"input": Port(data_config, self, True)}

    @property
    def output_ports(self) -> OutputPortDictionary:
        data_axis = FeatureAxis(variables=self.output_variable)
        data_config = DataConfiguration(None, [..., data_axis], feature_axis=data_axis)
        return {"output": Port(data_config, self)}
