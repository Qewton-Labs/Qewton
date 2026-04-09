from ..base import LayerNode, GraphNode, TrainableParameterNode
from ...config.configuration_base import DataConfiguration
from ..implementation import TorchImplementation, DEFAULT_DL_IMPLEMENTATION
from ...nodes.base import NodeState, InputPort, OutputPort
from ...optim.parameters.hyperparameter_base import HyperParameter
from .math import MatMul, Add
from ...pipelines.base import Graph


class TorchLinear(TorchImplementation):
    """Implementation of Linear layer in PyTorch backend."""

    def __init__(self, in_neurons, out_neurons, bias=True, **kwargs):
        from torch.nn import Linear as TLinear

        super().__init__(
            TLinear(in_features=in_neurons, out_features=out_neurons, bias=bias, **kwargs)
        )


class FunctionalLinear(GraphNode):

    def __init__(self, name="functional_linear", backend=DEFAULT_DL_IMPLEMENTATION):
        self.input = InputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="input",
        )
        self.weight = InputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="weight",
        )
        self.bias = InputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="bias",
            default=None,
        )
        self.output = OutputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="output",
        )
        self.backend = backend

        self.matmul_node = MatMul(backend=self.backend)
        self.add_node = Add(backend=self.backend)
        graph = Graph()
        graph.connect(self.matmul_node.output_ports[0], self.add_node.input_ports[0])

        super().__init__(
            graph=graph,
            input_ports={
                self.matmul_node.input_ports[0]: self.weight,
                self.matmul_node.input_ports[1]: self.input,
                self.add_node.input_ports[1]: self.bias,
            },
            output_ports={self.add_node.output_ports[0]: self.output},
            name=name,
        )


class Linear(GraphNode):
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
    ):
        self._input_ports[0].data_configuration.specify_dtype(backend)
        self._output_ports[0].data_configuration.specify_dtype(backend)
        self.in_neurons = HyperParameter.from_value(in_neurons, "In Neurons")
        self.out_neurons = HyperParameter.from_value(out_neurons, "Out Neurons")
        self.input = InputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="input",
        )
        self.weight = TrainableParameterNode(
            self.in_neurons, self.out_neurons, name="weight", backend=backend
        )
        if bias:
            self.bias = TrainableParameterNode(
                self.out_neurons, name="bias", backend=backend
            )
        # TODO: if this is automatically registred with the correct name in graphnode,
        # we can just pass a list
        self.output = OutputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="output",
        )

        self.functional_linear_node = FunctionalLinear(backend=backend)

        graph = Graph()
        graph.connect(self.weight, self.functional_linear_node.input)
        if bias:
            graph.connect(self.bias, self.functional_linear_node.bias)

        super().__init__(
            graph=graph,
            input_ports={self.functional_linear_node.input: self.input},
            output_ports={self.functional_linear_node.output: self.output},
            name=name,
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
        self._input_ports[0].data_configuration.specify_dtype(backend)
        self._output_ports[0].data_configuration.specify_dtype(backend)
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
