from ..base import LayerNode
from ..implementation import TorchImplementation, DEFAULT_DL_IMPLEMENTATION
from ...nodes.base import NodeState
from ...optim.parameters.hyperparameter_base import HyperParameter


class TorchLinear(TorchImplementation):
    """Implementation of Linear layer in PyTorch backend."""

    def __init__(self, in_neurons, out_neurons, bias=True, **kwargs):
        from torch.nn import Linear as TLinear

        super().__init__(
            TLinear(in_features=in_neurons, out_features=out_neurons, bias=bias, **kwargs)
        )


class Linear(LayerNode):
    """A node representing an activation function, which is a special type of
    algorithm that is applied element-wise to the input data.
    """

    existing_implementations = {TorchImplementation: TorchLinear}

    def __init__(
        self,
        in_neurons: int | HyperParameter,
        out_neurons: int | HyperParameter,
        bias=True,
        name="linear",
        backend=DEFAULT_DL_IMPLEMENTATION,
        **kwargs,
    ):

        super().__init__(name=name, backend=backend, state=NodeState.FIXED)
        self._input_ports[0].data_configuration.specify_backend(backend)
        self._output_ports[0].data_configuration.specify_backend(backend)
        self.in_neurons = HyperParameter.from_value(in_neurons, "In Neurons")
        self.out_neurons = HyperParameter.from_value(out_neurons, "Out Neurons")
        self.bias = bias
        self.kwargs = kwargs
        self.setup()

    def setup(self):
        self.implementation_instance = self.implementation(
            in_neurons=self.in_neurons.value,  # type: ignore
            out_neurons=self.out_neurons.value,  # type: ignore
            bias=self.bias,  # type: ignore
            **self.kwargs,
        )


class TorchConv2d(TorchImplementation):
    """Implementation of Conv2d layer in PyTorch backend."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs,
    ):
        from torch.nn import Conv2d as TConv2d

        super().__init__(
            TConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                **kwargs,
            )
        )


class Conv2d(LayerNode):
    """A node representing a 2D convolutional layer."""

    existing_implementations = {TorchImplementation: TorchConv2d}

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        name="conv2d",
        backend=DEFAULT_DL_IMPLEMENTATION,
        **kwargs,
    ):
        super().__init__(name=name, backend=backend, state=NodeState.FIXED)
        self._input_ports[0].data_configuration.specify_backend(backend)
        self._output_ports[0].data_configuration.specify_backend(backend)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.kwargs = kwargs
        self.setup()

    def setup(self):
        self.implementation_instance = self.implementation(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            **self.kwargs,
        )
