import warnings
from abc import abstractmethod
from enum import Enum, auto


from ..config.configuration_base import DataConfiguration
from ..config.variables import Variable
from ..config.axis import FeatureAxis
from ..nodes.base import Node, Port
from ..pipelines.base import Graph


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
    INCLUDES_IMAGINARY_VALUES = auto()  # Some optimizers do not work then


class AlgorithmNode(Node):
    """General node representing an algorithm that should solve a given problem
    or a part of it.
    """

    def __init__(
        self,
        name: str = "AlgorithmNode",
    ) -> None:
        """
        Args:
            input_variable (Variable): The input variables of the algorithm.
            output_variable (Variable): The output variables of the algorithm.
            name (str, optional): The name of the node. Defaults to "AlgorithmNode".
        """
        super().__init__(name=name)
        self._state: AlgorithmState = AlgorithmState.UNINITIALIZED

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

    @property
    def attributes(self) -> set[AlgorithmAttributes]:
        return set()

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

class GraphNode(Node):
    def __init__(self, graph: Graph, input_ports: list[Port], output_ports: list[Port], name: str = "GraphNode") -> None:
        super().__init__(name=name)
        self.graph = graph
        self.graph_input_ports = input_ports
        self.graph_output_ports = output_ports
    
    @property
    def input_ports(self) -> list[Port]:
        return [Port(data_configuration=port.data_configuration,
                     owner=self,
                     key=port.key,
                     is_required=port.required) for port in self.graph_input_ports]
    
    @property
    def output_ports(self) -> list[Port]:
        return [Port(data_configuration=port.data_configuration,
                     owner=self,
                     key=port.key,
                     is_required=port.required) for port in self.graph_output_ports]
    
    def _run(self, inputs: dict[str, any]) -> dict[str, any]:
        graph_runtime = self.graph.create_runtime()
        for port in self.graph_input_ports:
            graph_runtime.runtime_nodes[port.owner].receive(port.key, inputs[port.key])
        return graph_runtime.run(return_results=self.graph_output_ports)

class ActivationFunction(AlgorithmNode):
    """A node representing an activation function, which is a special type of
    algorithm that is applied element-wise to the input data.
    """

class DifferentiableNode(Node):
    """..."""

class TorchNode(DifferentiableNode):
    """A node representing a PyTorch module, which is a special type of
    algorithm that can be trained and run on a GPU.
    """
    def __init__()
        
    def setup(self) -> None:
        """Creates the underlying PyTorch module instance."""
        self._torch_module = self.create_torch_module()

    @property
    def torch_module(self):
        return self._torch_module
    
    def get_trainable_parameters(self) -> dict[str, any]:
        """Returns the trainable parameters of this node, which can be used for
        training the underlying algorithm (e.g. a neural network).

        Returns:
            dict[str, any]: A dictionary of trainable parameters.
        """
        return self.torch_module.parameters()
