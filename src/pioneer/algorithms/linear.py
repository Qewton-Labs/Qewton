from pioneer.config.backend import DEFAULT_DL_BACKEND, TorchBackend
from .base import LayerNode
from .implementation import TorchImplementation
from ..nodes.base import NodeState


class Linear(LayerNode):
    """A node representing an activation function, which is a special type of
    algorithm that is applied element-wise to the input data.
    """
    existing_implementations = {TorchBackend: TorchLinear}
    def __init__(
        self,
        in_neurons,
        out_neurons,
        bias=True,
        name="linear", backend=DEFAULT_DL_BACKEND, **kwargs
    ):

        super(Linear, self).__init__(name=name, backend=backend, state=NodeState.FIXED)
        self._input_ports[0].data_config.specify_backend(backend)
        self._output_ports[0].data_config.specify_backend(backend)
    
    def setup()


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
