from typing import Annotated, Literal, Generic

from qewton.algorithms.building_blocks.parameters import ParameterNode
from qewton.backends import DEFAULT_DL_BACKEND, TensorType, DeepLearningBackend
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.config.axes import (
    EllipsisDim,
    FeatureAxes,
    AxesDim,
    BatchAxes,
    GeometryAxes,
    EllipsisAxes,
)
from qewton.algorithms.building_blocks.activation_functions import ReLU
from qewton.optim.base import EvaluationPhase
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
        self.batch_axes = BatchAxes(AxesDim(None))
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
        self._graph.setup()
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
        kernel_size: int | tuple[int, ...] | HyperParameter,
        activation: type[Node] | HyperParameter = ReLU,
        bias: bool | HyperParameter = True,
        stride: int | HyperParameter | tuple[int, ...] = 1,
        padding: int | HyperParameter | tuple[int, ...] = 0,
        dilation: int | HyperParameter | tuple[int, ...] = 1,
        groups: int = 1,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        self.in_channels = HyperParameter.from_value(in_channels, "InChannels")
        self.out_channels = HyperParameter.from_value(out_channels, "OutChannels")
        self.kernel_size = HyperParameter.from_value(kernel_size, "Kernel")
        self.activation = HyperParameter.from_value(activation, "ActivationFunction")
        self.bias = HyperParameter.from_value(bias, "Bias")
        self.stride = HyperParameter.from_value(stride, "Stride")
        self.padding = HyperParameter.from_value(padding, "Padding")
        self.dilation = HyperParameter.from_value(dilation, "Dilation")
        self.groups = groups

        graph = self._build_network(backend=backend)
        super().__init__(
            graph=graph,
            input_ports=[self.conv1.input],
            output_ports=[self.activation2.output_ports[0]],
            backend=backend,
            **kwargs,
        )
        self._graph.setup()
        self.input = self.input_ports[0]
        self.output = self.output_ports[0]

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.activation,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        ]

    def _build_network(self, backend):
        dim = (
            1
            if isinstance(self.kernel_size.current_value, int)
            else len(self.kernel_size.current_value)
        )
        if dim not in (1, 2, 3):
            raise ValueError(f"Convolution are not implemented for dimension {dim}.")

        self.conv1 = Conv(
            self.in_channels.current_value,
            self.out_channels.current_value,
            self.kernel_size.current_value,
            self.bias.current_value,
            self.stride.current_value,
            self.padding.current_value,
            self.dilation.current_value,
            self.groups,
            backend,
        )
        self.batch_norm1 = BatchNorm(
            num_features=self.out_channels.current_value, dim=dim
        )
        self.activation1 = self.activation.current_value(
            name="activation1", backend=backend
        )
        self.conv2 = Conv(
            self.out_channels.current_value,
            self.out_channels.current_value,
            self.kernel_size.current_value,
            self.bias.current_value,
            self.stride.current_value,
            self.padding.current_value,
            self.dilation.current_value,
            self.groups,
            backend,
        )
        self.batch_norm2 = BatchNorm(
            num_features=self.out_channels.current_value, dim=dim
        )
        self.activation2 = self.activation.current_value(
            name="activation1", backend=backend
        )

        graph = Graph()
        graph.connect(self.conv1.output, self.batch_norm1)
        graph.connect(self.batch_norm1.output, self.activation1)
        graph.connect(self.activation1, self.conv2.input)
        graph.connect(self.conv2.output, self.batch_norm2)
        graph.connect(self.batch_norm2.output, self.activation2)

        return graph

    def setup(self) -> None:
        new_graph = self._build_network(self.backend)
        self.setup_graph(
            new_graph,
            input_ports=[self.conv1.input],
            output_ports=[self.activation2.output_ports[0]],
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
        self.backend: type[DeepLearningBackend[TensorType]] = backend
        self.kernel_size = kernel_size
        # Axes for dataconfigs:
        self.batch_axes = EllipsisAxes()
        self.feature_axes = FeatureAxes(shape=(AxesDim(None),))
        in_geo_dims, out_geo_dims = [], []
        for i in range(len(kernel_size)):
            in_geo_dims.append(AxesDim(None))
            out_geo_dims.append(self._build_output_dim(in_geo_dims[-1], i))
        self.in_geo_axes = GeometryAxes(shape=tuple(in_geo_dims))
        self.out_geo_axes = GeometryAxes(shape=tuple(out_geo_dims))
        super().__init__(name, state, backend)

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
class Interpolate(Node[TensorType]):
    """A node that interpolates a given tensor to a new shape.
    Expected inputs are 3-D, 4-D or 5-D in shape. E.g
    (batch, feature, width, [height, depth]), where the last to axis
    are optional.

    Args:
        size (int | tuple[int...] | None, optional):
            The output spatial size. Defaults to None.
        scale_factor (int  |  tuple[int...]  |  None, optional):
            A multiplier for the spatial size. The scale_factor has to fit the the
            number of spatial dimensions. Defaults to None.
            Either *size* or the *scale_factor* need to be provided.
        mode (Literal[ &quot;nearest&quot;, &quot;linear&quot;,
                       &quot;bilinear&quot;, &quot;bicubic&quot;,
                       &quot;trilinear&quot; ], optional):
            The type of interpolation scheme to use. Defaults to "nearest".
        align_corners (bool, optional): If the pixel data should be aligned along
            corners. Defaults to False.
        name (str, optional): Name of the node. Defaults to "InterpolateNode".
        backend (type[DeepLearningBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        size: int | tuple[int] | tuple[int, int] | tuple[int, int, int] | None = None,
        scale_factor: (
            int | tuple[int] | tuple[int, int] | tuple[int, int, int] | None
        ) = None,
        mode: Literal[
            "nearest", "linear", "bilinear", "bicubic", "trilinear"
        ] = "nearest",
        align_corners: bool | None = None,
        name: str = "InterpolateNode",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        self.size = (size,) if isinstance(size, int) else size
        self.scale_factor = (
            (scale_factor,) if isinstance(scale_factor, int) else scale_factor
        )
        self.interpolate_mode = mode
        self.align_corners = align_corners

        # Build the data config:
        self.batch_axes = BatchAxes(AxesDim(None))
        self.feature_axes = FeatureAxes(shape=(AxesDim(None),))
        # If a size is given the output shape is fix
        if self.size is not None:
            self.geo_axes_in = GeometryAxes(
                shape=tuple(AxesDim(None) for _ in range(len(self.size)))
            )
            self.geo_axes_out = GeometryAxes(shape=self.size)
        # For scaling we need to couple input and output
        elif self.scale_factor is not None:
            axes_dims = tuple(AxesDim(None) for _ in range(len(self.scale_factor)))
            self.geo_axes_in = GeometryAxes(shape=axes_dims)
            self.geo_axes_out = GeometryAxes(
                shape=tuple(a * s for a, s in zip(axes_dims, self.scale_factor))
            )
        else:
            ell_dim = EllipsisDim()
            self.geo_axes_in = GeometryAxes(shape=(ell_dim,))
            self.geo_axes_out = GeometryAxes(shape=(ell_dim,))

        super().__init__(name, NodeState.FIXED, backend)
        self.backend: DeepLearningBackend = self.backend
        self.input_ports[1].default = self.size

        self.input_port = self.input_ports[0]
        self.size_port = self.input_ports[1]

    def in_data_config(self):
        return DC(
            self.batch_axes,
            self.feature_axes,
            self.geo_axes_in,
            dtype=self.backend.default_dtype,
        )

    def out_data_config(self):
        return DC(
            self.batch_axes,
            self.feature_axes,
            self.geo_axes_out,
            dtype=self.backend.default_dtype,
        )

    def forward(
        self,
        x: Annotated[TensorType, in_data_config],
        size: Annotated[
            int | tuple[int] | tuple[int, int] | tuple[int, int, int] | None,
            FeatureAxes(shape=(AxesDim(None),)),
        ] = None,
    ) -> Annotated[TensorType, out_data_config]:
        if size is None and self.size is not None:
            size = self.size
        return self.backend.nn.interpolate(
            x,
            size=size,
            scale_factor=self.scale_factor,  # type: ignore
            mode=self.interpolate_mode,  # type: ignore
            align_corners=self.align_corners,
        )


# endregion


# region: BatchNorm
class FunctionalBatchNorm(Node[TensorType]):
    """A node implementing a batch normalization operation for 1D data.

    Args:
        dim (Literal[1, 2, 3]): The dimension of the input data.
        momentum (float, optional): The momentum used to update the running bias.
            Defaults to 0.1.
        eps (float, optional): A tolerance added to the variance normalization.
            Defaults to 1e-5.
        name (str, optional): Name of the node. Defaults to "BatchNorm1D".
        backend (type[DeepLearningBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        dim: Literal[1, 2, 3],
        momentum: float = 0.1,
        eps: float = 1e-5,
        name: str = "FunctionalBatchNorm",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        self.eps = eps
        self.momentum = momentum
        self.training = True
        if dim == 1:
            self.batch_norm_fn = backend.nn.batch_norm1d
        elif dim == 2:
            self.batch_norm_fn = backend.nn.batch_norm2d
        elif dim == 3:
            self.batch_norm_fn = backend.nn.batch_norm3d

        # Data configurations for the input and output ports
        self.feature_dim = AxesDim(None)
        self.batch_axes = BatchAxes(AxesDim(None))
        self.feature_axes = FeatureAxes(shape=(self.feature_dim,))
        self.geo_axes = GeometryAxes(shape=tuple(AxesDim(None) for _ in range(dim)))

        super().__init__(name, NodeState.INITIALIZED, backend)

    def set_mode(self, new_mode: EvaluationPhase):
        if new_mode == EvaluationPhase.TRAIN:
            self.training = True
        else:
            self.training = False
        return super().set_mode(new_mode)

    def data_config(self):
        return DC(
            self.batch_axes,
            self.feature_axes,
            self.geo_axes,
            dtype=self.backend.default_dtype,
        )

    def parameter_data_config(self):
        return DC(self.feature_axes, dtype=self.backend.default_dtype)

    def forward(
        self,
        x: Annotated[TensorType, data_config],
        running_mean: Annotated[TensorType, parameter_data_config],
        running_var: Annotated[TensorType, parameter_data_config],
        weight: Annotated[TensorType, parameter_data_config] = None,  # type: ignore
        bias: Annotated[TensorType, parameter_data_config] = None,  # type: ignore
    ) -> Annotated[TensorType, data_config]:
        return self.batch_norm_fn(
            x,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )


class BatchNorm(GraphNode, Generic[TensorType]):
    """A node implementing a batch normalization operation for 1D data.

    Args:
        num_features (int | HyperParameter): The number of features in the input data.
        dim (Literal[1, 2, 3]): The dimension of the input data.
        weight (bool, optional): If a trainable weight should be added. Defaults to False.
        bias (bool, optional): If a trainable bias should be added. Defaults to False.
        momentum (float, optional): The momentum used to update the running bias.
            Defaults to 0.1.
        eps (float, optional): A tolerance added to the variance normalization.
            Defaults to 1e-5.
        name (str, optional): Name of the node. Defaults to "BatchNorm".
        backend (type[DeepLearningBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        num_features: int | HyperParameter,
        dim: Literal[1, 2, 3],
        weight: bool = False,
        bias: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
        name: str = "BatchNorm",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        self.num_features = HyperParameter.from_value(num_features, "BatchNorm Feat.")
        self.dim = dim
        self.use_weight = weight
        self.use_bias = bias
        self.momentum = momentum
        self.eps = eps
        graph = self._build_graph(backend)

        super().__init__(
            graph=graph,
            input_ports=[self.functional_batch_norm.input_ports[0]],
            output_ports=[self.functional_batch_norm.output_ports[0]],
            backend=backend,
            name=name,
        )
        self._graph.setup()
        self.running_mean.fix_node_state()  # no automatic gradient tracking
        self.running_var.fix_node_state()
        self.input = self.input_ports[0]
        self.output = self.output_ports[0]

    def _build_graph(self, backend: type[DeepLearningBackend] = DEFAULT_DL_BACKEND):
        # Build all the nodes and the graph:
        graph = Graph()
        self.functional_batch_norm = FunctionalBatchNorm(
            dim=self.dim,  # type: ignore
            momentum=self.momentum,
            eps=self.eps,
            backend=backend,
        )
        self.running_mean = ParameterNode(
            (self.num_features.current_value,),
            initial_value=backend.math.zeros((self.num_features.current_value,)),
            name="running_mean",
            backend=backend,
        )
        self.running_var = ParameterNode(
            (self.num_features.current_value,),
            initial_value=backend.math.ones((self.num_features.current_value,)),
            name="running_var",
            backend=backend,
        )

        graph.connect(self.running_mean, self.functional_batch_norm.input_ports[1])
        graph.connect(self.running_var, self.functional_batch_norm.input_ports[2])
        # Add optional arguments
        if self.use_weight:
            self.weight = ParameterNode(
                (self.num_features.current_value,),
                initial_value=backend.math.ones((self.num_features.current_value,)),
                name="weight",
                backend=backend,
            )
            graph.connect(self.weight, self.functional_batch_norm.input_ports[3])
        if self.use_bias:
            self.bias = ParameterNode(
                (self.num_features.current_value,),
                initial_value=backend.math.zeros((self.num_features.current_value,)),
                name="bias",
                backend=backend,
            )
            graph.connect(self.bias, self.functional_batch_norm.input_ports[4])
        return graph

    def setup(self) -> None:
        new_graph = self._build_graph(self.backend)  # type: ignore
        self.setup_graph(
            new_graph,
            input_ports=[self.functional_batch_norm.input_ports[0]],
            output_ports=[self.functional_batch_norm.output_ports[0]],
        )
        self.running_mean.fix_node_state()  # no automatic gradient tracking
        self.running_var.fix_node_state()
        self.input = self.input_ports[0]
        self.output = self.output_ports[0]

    def forward(self, x):
        self.input.set_value(x)
        self.run()
        return self.output.value


class BatchNorm1D(BatchNorm[TensorType]):
    """A node implementing a batch normalization operation for 1D data.

    Args:
        num_features (int| HyperParameter): The number of features in the input data.
        weight (bool, optional): If a trainable weight should be added. Defaults to False.
        bias (bool, optional): If a trainable bias should be added. Defaults to False.
        momentum (float, optional): The momentum used to update the running bias.
            Defaults to 0.1.
        eps (float, optional): A tolerance added to the variance normalization.
            Defaults to 1e-5.
        name (str, optional): Name of the node. Defaults to "BatchNorm1D".
        backend (type[DeepLearningBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        num_features: int,
        weight: bool = False,
        bias: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
        name: str = "BatchNorm1D",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(
            num_features=num_features,
            dim=1,
            weight=weight,
            bias=bias,
            momentum=momentum,
            eps=eps,
            name=name,
            backend=backend,
        )


class BatchNorm2D(BatchNorm[TensorType]):
    """A node implementing a batch normalization operation for 2D data.

    Args:
        num_features (int | HyperParameter): The number of features in the input data.
        weight (bool, optional): If a trainable weight should be added. Defaults to False.
        bias (bool, optional): If a trainable bias should be added. Defaults to False.
        momentum (float, optional): The momentum used to update the running bias.
            Defaults to 0.1.
        eps (float, optional): A tolerance added to the variance normalization.
            Defaults to 1e-5.
        name (str, optional): Name of the node. Defaults to "BatchNorm2D".
        backend (type[DeepLearningBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        num_features: int,
        weight: bool = False,
        bias: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
        name: str = "BatchNorm2D",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(
            num_features=num_features,
            dim=2,
            weight=weight,
            bias=bias,
            momentum=momentum,
            eps=eps,
            name=name,
            backend=backend,
        )


class BatchNorm3D(BatchNorm[TensorType]):
    """A node implementing a batch normalization operation for 3D data.

    Args:
        num_features (int | HyperParameter): The number of features in the input data.
        weight (bool, optional): If a trainable weight should be added. Defaults to False.
        bias (bool, optional): If a trainable bias should be added. Defaults to False.
        momentum (float, optional): The momentum used to update the running bias.
            Defaults to 0.1.
        eps (float, optional): A tolerance added to the variance normalization.
            Defaults to 1e-5.
        name (str, optional): Name of the node. Defaults to "BatchNorm3D".
        backend (type[DeepLearningBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        num_features: int,
        weight: bool = False,
        bias: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
        name: str = "BatchNorm3D",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(
            num_features=num_features,
            dim=3,
            weight=weight,
            bias=bias,
            momentum=momentum,
            eps=eps,
            name=name,
            backend=backend,
        )


# endregion
