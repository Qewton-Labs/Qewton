from ..implementation import DEFAULT_DL_IMPLEMENTATION, Implementation
from ...optim.parameters.hyperparameter_base import HyperParameter
from ...config.configuration_base import DataConfiguration
from ...graphs.nodes import Node, NodeState, OutputPort


class TrainableParameterNode(Node):

    def __init__(
        self,
        shape: tuple[int | HyperParameter, ...],
        name: str = "Node",
        backend: Implementation = DEFAULT_DL_IMPLEMENTATION,
    ) -> None:
        super().__init__(name, state=NodeState.UNINITIALIZED)
        self.shape = tuple(
            HyperParameter.from_value(s, f"shape_{i}") for i, s in enumerate(shape)
        )
        self.backend = backend
        self.output = OutputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="parameters",
        )

    def setup(self) -> None:
        if self.state == NodeState.UNINITIALIZED:
            int_shape = tuple(hp.value for hp in self.shape)
            # TODO: We need some kind of initialization for these parameters
            # E.g. 0, rand, xavier,... But this also needs to be exposed to the outside
            params = self.backend.create_trainable_parameter(int_shape)
            self.output.set_value(params)
            self.state = NodeState.INITIALIZED

    def run(self) -> None:
        pass

    def reset(self):
        self.state = NodeState.UNINITIALIZED
