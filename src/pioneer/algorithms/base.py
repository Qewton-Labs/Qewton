from ..config.configuration_base import DataConfiguration
from ..graphs.nodes import InputPort, Node, NodeState, OutputPort
from .implementation import DEFAULT_DL_IMPLEMENTATION


# # TODO: Is this needed? Can we make this more natural?
# class AlgorithmAttributes(Enum):
#     SYMMETRIC = auto()  # if a "flipped" input yields the same output
#     TRANSLATION_INVARIANT = auto()
#     ROTATION_INVARIANT = auto()
#     LINEAR = auto()
#     DIFFERENTIABLE = auto()  # the output is differentiable in regards to the input
#     INVERTIBLE = auto()
#     NORMALIZES_DATA = auto()
#     DETERMINISTIC = auto()  # the run call (diffusion models for example not)
#     TRAINABLE = auto()  # TODO:Is this needed?
#     OUTPUTS_PROBABILITIES = auto()  # useful for classifiers?
#     GPU_ACCELERATED = auto()
#     MUTATES_INPUT = auto()  # if input is changed in-place
#     SUPPORTS_MISSING_VALUES = auto()  # if values like NaN are handled
#     INCLUDES_IMAGINARY_VALUES = auto()  # Some optimizers do not work then


class OperationNode(Node):
    """A node representing an operation, which is a type of algorithm that takes
    input data and produces output data without any trainable parameters.

    This class is built to easily wrap functions from backends.
    """

    implementations = {}

    def __init__(self, name=None, backend=DEFAULT_DL_IMPLEMENTATION):
        name = name if name is not None else self.__class__.__name__
        super().__init__(name=name, state=NodeState.FIXED)
        self.implementation = self.get_implementation(backend)

    def get_implementation(self, backend):
        if backend in self.implementations:
            return backend(*self.implementations[backend])
        raise ValueError(f"No implementation found for backend {backend}.")
