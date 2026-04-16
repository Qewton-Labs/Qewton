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

    args = {}
    outputs = []
    data_configs = {}
    implementations = []

    def __init__(self, name=None, backend=DEFAULT_DL_IMPLEMENTATION):
        self.implementation = self.get_implementation(backend)

        self.input_ports = [
            InputPort(
                data_configuration=self.data_configs.get(
                    name,
                    DataConfiguration.default_for_dtype(
                        self.implementation.standard_datatype()
                    ),
                ),
                node=self,
                name=name,
                default=arg,
            )
            for name, arg in self.args.items()
        ]
        self.output_ports = [
            OutputPort(
                data_configuration=self.data_configs.get(
                    name,
                    DataConfiguration.default_for_dtype(
                        self.implementation.standard_datatype()
                    ),
                ),
                node=self,
                name=name,
            )
            for name in self.outputs
        ]

        self._implementation_ordered_input_ports = (
            [self.get_input_port(name) for name in self.implementation.inputs]
            if self.implementation.inputs is not None
            else self.input_ports
        )
        self._implementation_ordered_output_ports = (
            [self.get_output_port(name) for name in self.implementation.outputs]
            if self.implementation.outputs is not None
            else self.output_ports
        )

        name = name if name is not None else self.__class__.__name__
        super().__init__(name=name, state=NodeState.FIXED)

    def get_implementation(self, backend):
        for impl in self.implementations:
            if isinstance(impl, backend):
                return impl
        raise ValueError(f"No implementation found for backend {backend}.")

    def run(self):
        outputs = self.implementation(
            *[port.value for port in self._implementation_ordered_input_ports]
        )
        for i, port in enumerate(self._implementation_ordered_output_ports):
            port.set_value(outputs[i])
