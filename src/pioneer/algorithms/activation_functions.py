from pioneer.config.backend import DEFAULT_DL_BACKEND, TorchBackend
from .base import LayerNode
from .implementation import TorchImplementation
from ..nodes.base import NodeState


class ActivationFunction(LayerNode):
    """A node representing an activation function, which is a special type of
    algorithm that is applied element-wise to the input data.
    """

    def __init__(self, name, backend=DEFAULT_DL_BACKEND):
        super(ActivationFunction, self).__init__(
            name=name, backend=backend, state=NodeState.FIXED
        )
        self._input_ports.set_backend(backend)
        self._output_ports.set_backend(backend)


class TorchReLU(TorchImplementation):
    """Implementation of ReLU Activation in PyTorch backend."""

    def __init__(self):
        from torch.nn import ReLU as TReLU

        super().__init__(TReLU)


class ReLU(ActivationFunction):
    """General ReLU Class."""

    existing_implementations = {TorchBackend: TorchReLU}

    def __init__(self, name="ReLU", backend=DEFAULT_DL_BACKEND):
        super(ReLU, self).__init__(name=name, backend=backend)


class TorchTanh(TorchImplementation):
    """Implementation of Tanh Activation in PyTorch backend."""

    def __init__(self):
        from torch.nn import Tanh as TTanh

        super().__init__(TTanh)


class Tanh(ActivationFunction):
    """General Tanh Class."""

    existing_implementations = {TorchBackend: TorchTanh}

    def __init__(self, name="Tanh", backend=DEFAULT_DL_BACKEND):
        super(Tanh, self).__init__(name=name, backend=backend)
