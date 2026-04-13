from enum import Enum, auto

from pioneer.optim.parameters.hyperparameter_base import HyperParameter

from ..config.configuration_base import DataConfiguration
from ..graphs.nodes import InputPort, Node, NodeState, OutputPort
from ..optim.parameters.trainable_parameters import TrainableParameters
from .implementation import DEFAULT_DL_IMPLEMENTATION, Implementation


# TODO: Is this needed? Can we make this more natural?
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
    INCLUDES_IMAGINARY_VALUES = auto()  # Some optimizers do not work then


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


class LayerNode(Node):
    """A node representing a unary operation, which is an algorithm that takes
    one input and produces one output. Uses a specific implementation of the
    algorithm by calling the respective backend.
    """

    existing_implementations: dict[type[Implementation], type[Implementation]]

    def __init__(
        self,
        name: str = "LayerNode",
        backend: Implementation = DEFAULT_DL_IMPLEMENTATION,
        state: NodeState = NodeState.FIXED,
    ):
        """Initializes a Layer node with a single input and output port.

        Args:
            name (str, optional): The name of this node. Defaults to "LayerNode".
            implementation (callable, optional): The function that implements the
                unary operation. It should take one argument (the input) and
                return the output. Defaults to None.
            state (NodeState, optional): The initial state of this node.
                Defaults to NodeState.FIXED.
        """
        super().__init__(name=name, state=state)
        self.backend = backend
        self.implementation = self.set_implementation()
        self.implementation_instance: Implementation
        self._input_ports: list[InputPort] = [  # type: ignore
            InputPort(DataConfiguration([]), self, "input")
        ]
        self._output_ports: list[OutputPort] = [  # type: ignore
            OutputPort(DataConfiguration([]), self, "output")
        ]

    def run(self):
        self._output_ports[0].set_value(
            self.implementation_instance(self._input_ports[0].value)
        )

    def set_implementation(self):
        if type(self).existing_implementations is None:
            raise NotImplementedError(f"No implementations defined for {self.name}.")
        if self.backend not in type(self).existing_implementations:
            raise NotImplementedError(
                f"Backend {self.backend} is not supported for {self.name}."
            )
        return type(self).existing_implementations[self.backend]

    @property
    def trainable_parameters(self):
        return TrainableParameters(self.node_id, self.implementation.trainable_parameters)
