"""Skip connections of Fourier Neural Operator blocks."""

from typing import Annotated, Generic, Literal

from qewton.algorithms.building_blocks.conv import Flattened1dConv
from qewton.algorithms.building_blocks.creation import Identity
from qewton.algorithms.building_blocks.math import Add, Multiply
from qewton.algorithms.building_blocks.parameters import ParameterNode
from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.config.data_configurations import DataConfiguration
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node

SkipType = Literal["soft-gating", "linear", "identity"]
"""The kinds of skip connections available in `skip_connection`."""


class SoftGating(GraphNode, Generic[TensorType]):
    """Soft-gating skip connection.

    Weights every channel of the input with a learned scalar, `x * w (+ b)`,
    where `w` (and `b`) have the shape `(1, channels, 1, ..., 1)`. Both the weight and
    the bias are initialized with ones.

    Attributes:
        weight (ParameterNode): The learned per-channel scale.
        bias (ParameterNode | None): The learned per-channel offset, or None if the
            gating has no bias.

    Args:
        in_features (int): Number of channels of the input.
        out_features (int | None, optional): Number of channels of the output. Only
            there for compatibility with the other skip connections; it has to be
            None or equal to `in_features`. Defaults to None.
        n_dim (int, optional): Number of spatial dimensions of the input (excluding
            batch and channels). Defaults to 2.
        bias (bool, optional): If a learned bias should be added. Defaults to False.
        name (str, optional): Name of the node. Defaults to "soft_gating".
        backend (type[Backend[TensorType]], optional): The backend used for the
            computations. Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        n_dim: int = 2,
        bias: bool = False,
        name: str = "soft_gating",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}, "
                "but these two must be the same for soft-gating."
            )
        parameter_shape = (1, in_features, *(1,) * n_dim)

        self.weight = ParameterNode(
            parameter_shape,
            initial_value=backend.math.ones(parameter_shape),
            name="weight",
            backend=backend,
        )
        self.multiply = Multiply(backend=backend)
        graph = Graph()
        graph.connect(self.weight.output_ports[0], self.multiply.input_ports[1])
        output_port = self.multiply.output_ports[0]

        if bias:
            self.bias = ParameterNode(
                parameter_shape,
                initial_value=backend.math.ones(parameter_shape),
                name="bias",
                backend=backend,
            )
            self.add = Add(backend=backend)
            graph.connect(self.multiply.output_ports[0], self.add.input_ports[0])
            graph.connect(self.bias.output_ports[0], self.add.input_ports[1])
            output_port = self.add.output_ports[0]
        else:
            self.bias = None

        super().__init__(
            graph=graph,
            input_ports=[self.multiply.input_ports[0]],
            output_ports=[output_port],
            name=name,
            backend=backend,
        )
        self._graph.setup()

    def forward(
        self, x: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        """Scales (and shifts) every channel of the input.

        The weight is broadcast over batch and space, which the data configuration
        cannot express, so the input and output configuration is left flexible.

        Args:
            x (TensorType): Input of shape `(batch, in_features, *spatial_dims)`.

        Returns:
            TensorType: The gated input, same shape as `x`.
        """
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value  # type: ignore


def skip_connection(
    in_features: int,
    out_features: int,
    n_dim: int = 2,
    bias: bool = False,
    skip_type: SkipType = "soft-gating",
    backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
) -> Node:
    """Builds a skip connection node.

    Args:
        in_features (int): Number of channels of the input.
        out_features (int): Number of channels of the output.
        n_dim (int, optional): Number of spatial dimensions of the input. Only used
            by the soft-gating. Defaults to 2.
        bias (bool, optional): If the soft-gating or the linear skip should have a
            bias. Not used by the identity. Defaults to False.
        skip_type (SkipType, optional): "soft-gating" for a learned scale per channel,
            "linear" for a 1x1 convolution, or "identity". Defaults to "soft-gating".
        backend (type[Backend[TensorType]], optional): The backend used for the
            computations. Defaults to DEFAULT_DL_BACKEND.

    Returns:
        Node: The skip connection, with a single input and a single output port.

    Raises:
        ValueError: If `skip_type` is unknown, or if the soft-gating or the identity
            is requested with `in_features != out_features`.
    """
    if skip_type == "soft-gating":
        return SoftGating(
            in_features, out_features, n_dim=n_dim, bias=bias, backend=backend
        )
    if skip_type == "linear":
        return Flattened1dConv(
            in_features, out_features, kernel_size=1, bias=bias, backend=backend
        )
    if skip_type == "identity":
        if in_features != out_features:
            raise ValueError(
                "Identity skip connection needs in_features == out_features, "
                f"got {in_features} and {out_features}."
            )
        return Identity(backend=backend)
    raise ValueError(f"Unexpected skip_type {skip_type}.")
