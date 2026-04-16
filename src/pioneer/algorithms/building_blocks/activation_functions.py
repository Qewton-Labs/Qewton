from ..base import OperationNode
from ...graphs.nodes import NO_DEFAULT
from ..implementation import (
    TorchImplementation,
    TensorflowImplementation,
)


class ReLU(OperationNode):
    """General ReLU Class."""

    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("relu",),
        TensorflowImplementation: ("relu",),
    }


class Tanh(OperationNode):
    """General Tanh Class."""

    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("tanh",),
        TensorflowImplementation: ("tanh",),
    }


class Sigmoid(OperationNode):
    """General Sigmoid Class."""

    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("sigmoid",),
        TensorflowImplementation: ("sigmoid",),
    }
