from typing import Annotated, Literal, Generic

from qewton.algorithms.building_blocks.parameters import ParameterNode
from qewton.backends import DEFAULT_DL_BACKEND, TensorType, DeepLearningBackend
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.config.axes import FeatureAxes, AxesDim, BatchAxes, GeometryAxes
from qewton.algorithms.building_blocks.activation_functions import ReLU
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node, NodeState
from qewton.graphs.control_nodes.graph_node import GraphNode

# region: Convolutions


class FunctionalConv(Node[TensorType]):
    """A functional convolution layer, using the same arguments as
    the convolution node.

    """

    def __init__(
        self,
        dim: Literal[1, 2, 3],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        name: str = "FunctionalConvolution",
        state: NodeState = NodeState.FIXED,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        self.backend: type[DeepLearningBackend[TensorType]] = backend
        if dim == 1:
            self.conv_fn = self.backend.nn.conv1d
        elif dim == 2:
            self.conv_fn = self.backend.nn.conv2d
        elif dim == 3:
            self.conv_fn = self.backend.nn.conv3d
        self.stride = stride if isinstance(stride, tuple) else (stride,) * dim
        self.padding = padding if isinstance(padding, tuple) else (padding,) * dim
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * dim
        self.groups = groups

        # Shape information for data configurations
        self.batch_axes = BatchAxes()
        channel_dim_in = AxesDim(None)
        self.channel_dim_out = AxesDim(None)
        kernel_dims = tuple(AxesDim(None) for _ in range(dim))
        self.feature_axes_x = FeatureAxes(shape=(channel_dim_in,))
        self.feature_axes_weight = FeatureAxes(
            shape=(self.channel_dim_out, channel_dim_in) + kernel_dims
        )
        geo_shape = tuple(AxesDim(None) for _ in range(dim))
        self.geo_axes = GeometryAxes(shape=geo_shape)
        out_shape = []
        for i, in_dim in enumerate(geo_shape):
            out_dim = (
                in_dim + 2 * self.padding[i] - self.dilation[i] * (kernel_dims[i] - 1) - 1
            )
            out_dim /= self.stride[i]
            out_dim += 1
            out_shape.append(out_dim)
        self.geo_axes_out = GeometryAxes(shape=tuple(out_shape))

        # Register the ports as own variables
        super().__init__(name, state, backend)
        self.input = self.input_ports[0]
        self.kernel = self.input_ports[1]
        self.bias = self.input_ports[2]
        self.output = self.output_ports[0]

    def x_data_config(self):
        return DC(
            self.batch_axes,
            self.feature_axes_x,
            self.geo_axes,
            dtype=self.backend.default_dtype,
        )

    def weight_data_config(self):
        return DC(self.feature_axes_weight, dtype=self.backend.default_dtype)

    def bias_data_config(self):
        return DC(
            FeatureAxes(shape=(self.channel_dim_out,)), dtype=self.backend.default_dtype
        )

    def out_data_config(self):
        return DC(
            self.batch_axes,
            FeatureAxes(shape=(self.channel_dim_out,)),
            self.geo_axes_out,
            dtype=self.backend.default_dtype,
        )

    def forward(
        self,
        x: Annotated[TensorType, x_data_config],
        weight: Annotated[TensorType, weight_data_config],
        bias: Annotated[TensorType, bias_data_config] = None,  # type: ignore
    ) -> Annotated[TensorType, out_data_config]:
        return self.conv_fn(
            x,
            weight=weight,
            bias=bias,
            stride=self.stride,  # type: ignore
            padding=self.padding,  # type: ignore
            dilation=self.dilation,  # type: ignore
            groups=self.groups,
        )


class Conv(GraphNode, Generic[TensorType]):
    """A node implementing a convolution operation. The dimension of the
    convolution is automatically determined from the specified kernel
    size.

    Args:
        in_channels (int | HyperParameter): The number of input channels.
        out_channels (int | HyperParameter): The number of output channels.
        kernel_size (int | tuple[int  |  HyperParameter, ...] | HyperParameter):
            The size of the convolution kernel. The dimension of the convolution
            is determined by the length of the tuple (or set to 1 in case of an
            integer).
        bias (bool, optional): If a bias should be included. Defaults to True.
        stride (int | tuple[int, ...], optional): Controls the stride for the
            cross-correlation. Defaults to 1.
        padding (int | tuple[int, ...], optional): The amount of padding applied
            to each axis of the input. Defaults to 0.
        dilation (int | tuple[int, ...], optional): The spacing between kernel
            points. Defaults to 1, so direct neighbors are taken.
        groups (int, optional): The connections between inputs and outputs.
            Both input and output channels are divided into groups, and convolutions are
            applied separately to each group. The output is the concatenation of all
            the groups. Defaults to 1.
        backend (type[DeepLearningBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        in_channels: int | HyperParameter,
        out_channels: int | HyperParameter,
        kernel_size: int | tuple[int | HyperParameter, ...] | HyperParameter,
        bias: bool = True,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        if isinstance(kernel_size, int) or isinstance(kernel_size, HyperParameter):
            kernel_size = (kernel_size,)

        self.kernel = ParameterNode(
            (out_channels, in_channels) + kernel_size,
            name="weight_kernel",
            backend=backend,
        )
        if bias:
            self.bias = ParameterNode((out_channels,), name="bias", backend=backend)

        assert len(kernel_size) in [
            1,
            2,
            3,
        ], "Only 1D, 2D, and 3D convolutions are supported."
        self.conv_node = FunctionalConv(
            dim=len(kernel_size),  # type: ignore
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            name="functional_conv1d",
            backend=backend,
        )

        graph = Graph()
        graph.connect(self.kernel, self.conv_node.kernel)
        if bias:
            graph.connect(self.bias, self.conv_node.bias)

        super().__init__(
            graph=graph,
            input_ports=[self.conv_node.input],
            output_ports=[self.conv_node.output],
            backend=backend,
            **kwargs,
        )
        self.input = self.input_ports[0]
        self.output = self.output_ports[0]

    def forward(self, x):
        self.input.set_value(x)
        self.run()
        return self.output.value


class Conv1D(Conv[TensorType]):

    def __init__(
        self,
        in_channels: int | HyperParameter,
        out_channels: int | HyperParameter,
        kernel_size: int | HyperParameter | tuple[int | HyperParameter],
        bias: bool = True,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            bias,
            stride,
            padding,
            dilation,
            groups,
            backend,
            **kwargs,
        )


class Conv2D(Conv[TensorType]):

    def __init__(
        self,
        in_channels: int | HyperParameter,
        out_channels: int | HyperParameter,
        kernel_size: (
            int | HyperParameter | tuple[int | HyperParameter, int | HyperParameter]
        ),
        bias: bool = True,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        if isinstance(kernel_size, int) or isinstance(kernel_size, HyperParameter):
            kernel_size = (kernel_size, kernel_size)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            bias,
            stride,
            padding,
            dilation,
            groups,
            backend,
            **kwargs,
        )


class Conv3D(Conv[TensorType]):

    def __init__(
        self,
        in_channels: int | HyperParameter,
        out_channels: int | HyperParameter,
        kernel_size: (
            int
            | HyperParameter
            | tuple[int | HyperParameter, int | HyperParameter, int | HyperParameter]
        ),
        bias: bool = True,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        if isinstance(kernel_size, int) or isinstance(kernel_size, HyperParameter):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            bias,
            stride,
            padding,
            dilation,
            groups,
            backend,
            **kwargs,
        )


class DoubleConv(GraphNode, Generic[TensorType]):
    """A node implementing two convolution operations in sequence. The dimension of the
    convolution is automatically determined from the specified kernel
    size.

    Args:
        in_channels (int | HyperParameter): The number of input channels.
        out_channels (int | HyperParameter): The number of output channels.
        kernel_size (int | tuple[int  |  HyperParameter, ...] | HyperParameter):
            The size of the convolution kernel. The dimension of the convolution
            is determined by the length of the tuple (or set to 1 in case of an
            integer).
        activation: type[Node] | HyperParameter = ReLU,
        bias (bool, optional): If a bias should be included. Defaults to True.
        stride (int | tuple[int, ...], optional): Controls the stride for the
            cross-correlation. Defaults to 1.
        padding (int | tuple[int, ...], optional): The amount of padding applied
            to each axis of the input. Defaults to 0.
        dilation (int | tuple[int, ...], optional): The spacing between kernel
            points. Defaults to 1, so direct neighbors are taken.
        groups (int, optional): The connections between inputs and outputs.
            Both input and output channels are divided into groups, and convolutions are
            applied separately to each group. The output is the concatenation of all
            the groups. Defaults to 1.
        backend (type[DeepLearningBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        in_channels: int | HyperParameter,
        out_channels: int | HyperParameter,
        kernel_size: int | tuple[int | HyperParameter, ...] | HyperParameter,
        activation: type[Node] = ReLU,
        bias: bool = True,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        self.conv1 = Conv(
            in_channels,
            out_channels,
            kernel_size,
            bias,
            stride,
            padding,
            dilation,
            groups,
            backend,
        )
        self.activation1 = activation(name="activation1", backend=backend)
        self.conv2 = Conv(
            out_channels,
            out_channels,
            kernel_size,
            bias,
            stride,
            padding,
            dilation,
            groups,
            backend,
        )
        self.activation2 = activation(name="activation1", backend=backend)

        graph = Graph()
        graph.connect(self.conv1.output, self.activation1)
        graph.connect(self.activation1, self.conv2.input)
        graph.connect(self.conv2.output, self.activation2)

        super().__init__(
            graph=graph,
            input_ports=[self.conv1.input],
            output_ports=[self.activation2.output_ports[0]],
            backend=backend,
            **kwargs,
        )

        self.input = self.input_ports[0]
        self.output = self.output_ports[0]

    def forward(self, x):
        self.input.set_value(x)
        self.run()
        return self.output.value


# endregion


# region: Pooling
class PoolingNode(Node[TensorType]):
    """Helper node to combine some syntax that appears in every pooling node."""

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        name: str | None = None,
        state: NodeState = NodeState.FIXED,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(name, state, backend)
        self.backend: type[DeepLearningBackend[TensorType]] = backend
        self.kernel_size = kernel_size
        # Axes for dataconfigs:
        self.batch_axes = BatchAxes()
        self.feature_axes = FeatureAxes(shape=(AxesDim(None),))
        in_geo_dims, out_geo_dims = [], []
        for i in range(len(kernel_size)):
            in_geo_dims.append(AxesDim(None))
            out_geo_dims.append(self._build_output_dim(in_geo_dims[-1], i))
        self.in_geo_axes = GeometryAxes(shape=tuple(in_geo_dims))
        self.out_geo_axes = GeometryAxes(shape=tuple(out_geo_dims))

    def _build_output_dim(self, input_dim, dim_idx) -> AxesDim:
        """Builds the coupling between the input and output axes, which
        is influenced by this pooling operation."""
        raise NotImplementedError

    def in_data_config(self):
        return DC(
            self.batch_axes,
            self.feature_axes,
            self.in_geo_axes,
            dtype=self.backend.default_dtype,
        )

    def out_data_config(self):
        return DC(
            self.batch_axes,
            self.feature_axes,
            self.out_geo_axes,
            dtype=self.backend.default_dtype,
        )


class MaxPool1D(PoolingNode[TensorType]):
    """A pooling node, which reduces the size of the input by taking the maximum
    value in each kernel window.

    Args:
        kernel_size (int | tuple[int, ...]): The size of the pooling kernel.
        stride (int | tuple[int, ...] | None, optional): Stride for pooling.
            Defaults to None.
        padding (int | tuple[int, ...], optional): Additionally padding at the boundary.
            Defaults to 0.
        dilation (int | tuple[int, ...], optional): Dilation for the pooling window.
            Defaults to 1.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] | None = None,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        name: str | None = "MaxPooling",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        kernel_size = self._pack_tuple(kernel_size)
        if stride is None:
            stride = kernel_size
        self.stride = self._pack_tuple(stride)
        self.padding = self._pack_tuple(padding)
        self.dilation = self._pack_tuple(dilation)
        super().__init__(kernel_size, name, NodeState.FIXED, backend)

    def _build_output_dim(self, input_dim, dim_idx) -> AxesDim:
        out_dim = (
            input_dim
            + 2 * self.padding[dim_idx]
            - self.dilation[dim_idx] * (self.kernel_size[dim_idx] - 1)
            - 1
        )
        out_dim /= self.stride[dim_idx]
        return out_dim + 1

    def _pack_tuple(self, data: tuple[int, ...] | int) -> tuple[int]:
        return (data[0],) if isinstance(data, tuple) else (data,)

    def in_data_config(self):  # pylint: disable=W0246
        return super().in_data_config()

    def out_data_config(self):  # pylint: disable=W0246
        return super().out_data_config()

    def forward(
        self, x: Annotated[TensorType, in_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        return self.backend.nn.max_pool1d(
            x, self.kernel_size, self.stride, self.padding, self.dilation  # type: ignore
        )


class MaxPool2D(MaxPool1D[TensorType]):
    def _pack_tuple(self, data: tuple[int, ...] | int) -> tuple[int, int]:
        return (data[0], data[1]) if isinstance(data, tuple) else (data, data)

    def in_data_config(self):  # pylint: disable=W0246
        return super().in_data_config()

    def out_data_config(self):  # pylint: disable=W0246
        return super().out_data_config()

    def forward(
        self, x: Annotated[TensorType, in_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        return self.backend.nn.max_pool2d(
            x,
            self.kernel_size,  # type: ignore
            self.stride,  # type: ignore
            self.padding,  # type: ignore
            self.dilation,  # type: ignore
        )


class MaxPool3D(MaxPool1D[TensorType]):

    def _pack_tuple(self, data: tuple[int, ...] | int) -> tuple[int, int, int]:
        return (
            (data[0], data[1], data[2]) if isinstance(data, tuple) else (data, data, data)
        )

    def in_data_config(self):  # pylint: disable=W0246
        return super().in_data_config()

    def out_data_config(self):  # pylint: disable=W0246
        return super().out_data_config()

    def forward(
        self, x: Annotated[TensorType, in_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        return self.backend.nn.max_pool3d(
            x,
            self.kernel_size,  # type: ignore
            self.stride,  # type: ignore
            self.padding,  # type: ignore
            self.dilation,  # type: ignore
        )


class AvgPool1D(PoolingNode[TensorType]):
    """A pooling node, which reduces the size of the input by taking the average
    value in the kernel window.

    Args:
        kernel_size (int | tuple[int, ...]): The size of the pooling kernel.
        stride (int | tuple[int, ...] | None, optional): Stride for pooling.
            Defaults to None.
        padding (int | tuple[int, ...], optional): Additionally padding at the boundary.
            Defaults to 0.
        dilation (int | tuple[int, ...], optional): Dilation for the pooling window.
            Defaults to 1.
        count_include_pad (bool, optional): Whether to include the padding in the
            average calculation. Defaults to True.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] | None = None,
        padding: int | tuple[int, ...] = 0,
        count_include_pad: bool = True,
        name: str | None = "AvgPooling",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        kernel_size = self._pack_tuple(kernel_size)
        if stride is None:
            stride = kernel_size
        self.stride = self._pack_tuple(stride)
        self.padding = self._pack_tuple(padding)
        self.count_include_pad = count_include_pad
        super().__init__(kernel_size, name, NodeState.FIXED, backend)

    def _build_output_dim(self, input_dim, dim_idx) -> AxesDim:
        out_dim = input_dim + 2 * self.padding[dim_idx] - self.kernel_size[dim_idx] - 1
        out_dim /= self.stride[dim_idx]
        return out_dim + 1

    def _pack_tuple(self, data: tuple[int, ...] | int) -> tuple[int]:
        return (data[0],) if isinstance(data, tuple) else (data,)

    def in_data_config(self):  # pylint: disable=W0246
        return super().in_data_config()

    def out_data_config(self):  # pylint: disable=W0246
        return super().out_data_config()

    def forward(
        self, x: Annotated[TensorType, in_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        return self.backend.nn.avg_pool1d(
            x,
            self.kernel_size,  # type: ignore
            stride=self.stride[0],
            padding=self.padding[0],
            count_include_pad=self.count_include_pad,
        )


class AvgPool2D(AvgPool1D[TensorType]):

    def _pack_tuple(self, data: tuple[int, ...] | int) -> tuple[int, int]:
        return (data[0], data[1]) if isinstance(data, tuple) else (data, data)

    def in_data_config(self):  # pylint: disable=W0246
        return super().in_data_config()

    def out_data_config(self):  # pylint: disable=W0246
        return super().out_data_config()

    def forward(
        self, x: Annotated[TensorType, in_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        return self.backend.nn.avg_pool2d(
            x,
            self.kernel_size,  # type: ignore
            self.stride,  # type: ignore
            self.padding,  # type: ignore
            count_include_pad=self.count_include_pad,
        )


class AvgPool3D(AvgPool1D[TensorType]):

    def _pack_tuple(self, data: tuple[int, ...] | int) -> tuple[int, int, int]:
        return (
            (data[0], data[1], data[2]) if isinstance(data, tuple) else (data, data, data)
        )

    def in_data_config(self):  # pylint: disable=W0246
        return super().in_data_config()

    def out_data_config(self):  # pylint: disable=W0246
        return super().out_data_config()

    def forward(
        self, x: Annotated[TensorType, in_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        return self.backend.nn.avg_pool3d(
            x,
            self.kernel_size,  # type: ignore
            self.stride,  # type: ignore
            self.padding,  # type: ignore
            count_include_pad=self.count_include_pad,
        )


# endregion


# region: Upsampling


# endregion


# region: BatchNorm
# TODO: Add state dependent evaluation!
