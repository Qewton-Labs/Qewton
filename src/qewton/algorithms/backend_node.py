from __future__ import annotations
from typing import Generic

from qewton.backends.base import DeepLearningBackend
from qewton.graphs.nodes import Node, NodeState
from qewton.backends import (
    DEFAULT_DL_BACKEND,
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
    """A node that has a DeepLearningBackend.
    Can be used e.g. for all basic building blocks to define only the forward method.
    """

    def __init__(
        self,
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        name = name if name is not None else self.__class__.__name__
        super().__init__(name=name, state=NodeState.FIXED, backend=backend)
        self.backend: type[DeepLearningBackend[TensorType]] = self.backend

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "The forward method must be implemented by subclasses of BackendNode."
        )
