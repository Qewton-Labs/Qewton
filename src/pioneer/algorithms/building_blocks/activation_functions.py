from ..base import LayerNode
from ..implementation import DEFAULT_DL_IMPLEMENTATION, TorchImplementation
from ...graphs.nodes import NodeState


class ActivationFunction(LayerNode):
    """A node representing an activation function, which is a special type of
    algorithm that is applied element-wise to the input data.
    """

    def __init__(self, name="Activation Functions", backend=DEFAULT_DL_IMPLEMENTATION):
        super().__init__(name=name, backend=backend, state=NodeState.FIXED)
        self._input_ports[0].data_configuration.specify_dtype(backend)
        self._output_ports[0].data_configuration.specify_dtype(backend)
        self.implementation_instance = self.implementation()


class TorchReLU(TorchImplementation):
    """Implementation of ReLU Activation in PyTorch backend."""

    def __init__(self):
        from torch.nn import ReLU as TReLU

        super().__init__(TReLU())


class ReLU(ActivationFunction):
    """General ReLU Class."""

    existing_implementations = {TorchImplementation: TorchReLU}

    def __init__(self, name="ReLU", backend=DEFAULT_DL_IMPLEMENTATION):
        super().__init__(name=name, backend=backend)


class TorchTanh(TorchImplementation):
    """Implementation of Tanh Activation in PyTorch backend."""

    def __init__(self):
        from torch.nn import Tanh as TTanh

        super().__init__(TTanh())


class Tanh(ActivationFunction):
    """General Tanh Class."""

    existing_implementations = {TorchImplementation: TorchTanh}

    def __init__(self, name="Tanh", backend=DEFAULT_DL_IMPLEMENTATION):
        super().__init__(name=name, backend=backend)


class TorchSigmoid(TorchImplementation):
    """Implementation of Sigmoid Activation in PyTorch backend."""

    def __init__(self):
        from torch.nn import Sigmoid as TSigmoid

        super().__init__(TSigmoid())


class Sigmoid(ActivationFunction):
    """General Sigmoid Class."""

    existing_implementations = {TorchImplementation: TorchSigmoid}

    def __init__(self, name="Sigmoid", backend=DEFAULT_DL_IMPLEMENTATION):
        super().__init__(name=name, backend=backend)
