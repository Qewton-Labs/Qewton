from __future__ import annotations
from typing import Generic

from ..graphs.nodes import Node, NodeState
from ..config.backend import (
    DEFAULT_DL_BACKEND,
    Backend,
    TorchBackend,
    TensorflowBackend,
    TensorType,
)

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


class BackendNode(Node, Generic[TensorType]):
    """A node representing an operation, which is a type of algorithm that takes
    input data and produces output data without any trainable parameters.

    This class is built to easily wrap functions from backends.
    """

    def __init__(
        self, name=None, backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND
    ):
        name = name if name is not None else self.__class__.__name__
        super().__init__(name=name, state=NodeState.FIXED, backend=backend)
        _ = backend.import_library()

        self.choose_implementation()

    def choose_implementation(self):
        subclass_methods = {
            TorchBackend: self.torch_implementation,
            TensorflowBackend: self.tensorflow_implementation,
        }

        parent_methods = {
            TorchBackend: BackendNode.torch_implementation,
            TensorflowBackend: BackendNode.tensorflow_implementation,
        }

        # check whether the subclass overrides the method
        if subclass_methods[self.backend] is not parent_methods[self.backend]:
            self.implementation = subclass_methods[self.backend]
        else:
            if self.default_implementation is BackendNode.default_implementation:
                self.implementation = self.default_implementation
            else:
                assert (
                    self.forward is not BackendNode.forward
                ), "If no specific implementation is provided, the forward method must\
                    be overridden."

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "The forward method must be implemented by subclasses of BackendNode."
        )

    def default_implementation(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have a default implementation."
        )

    def torch_implementation(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have a Torch implementation."
        )

    def tensorflow_implementation(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have a Tensorflow implementation."
        )
