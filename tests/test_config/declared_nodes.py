"""Nodes that only declare their ports, for tests of configuration matching.

`Declared` takes the configurations of its ports directly; objects shared
between them are shared by the ports, as in the annotations of a real
node; `pass_through` is one that passes any data on. `B`, `G` and `F`
build batch, geometry and feature axes of one dimension; `dc` builds a
configuration of them, `unknown` one of an ellipsis only.
"""

from typing import Any

from qewton.config.axes import BatchAxes, EllipsisAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import InputPort, Node, OutputPort


class Declared(Node):
    """A node with the given port configurations; it computes nothing.

    Args:
        name: The name of the node.
        inputs: The configuration of each input port, in order.
        outputs: The configuration of each output port, in order.
    """

    def __init__(
        self,
        name: str,
        inputs: tuple[DataConfiguration, ...] = (),
        outputs: tuple[DataConfiguration, ...] = (),
    ) -> None:
        super().__init__(name=name)
        self._input_ports = [
            InputPort(config, node=self, name=f"in_{i}") for i, config in enumerate(inputs)
        ]
        self._output_ports = [
            OutputPort(config, node=self, name=f"out_{i}") for i, config in enumerate(outputs)
        ]

    def forward(self, *inputs: Any) -> Any:
        raise NotImplementedError("Declared nodes only declare their ports.")


def pass_through(name: str) -> Declared:
    """A node passing any data on unchanged."""
    e = EllipsisAxes()
    return Declared(name, inputs=(DataConfiguration(e),), outputs=(DataConfiguration(e),))


def B(size: int | None = None) -> BatchAxes:
    return BatchAxes(size)


def G(size: int | None = None) -> GeometryAxes:
    return GeometryAxes(shape=(size,))


def F(size: int | None = None) -> FeatureAxes:
    return FeatureAxes(shape=(size,))


def dc(*axes) -> DataConfiguration:
    return DataConfiguration(*axes, dtype=Any)


def unknown() -> DataConfiguration:
    return DataConfiguration(EllipsisAxes(), dtype=Any)


def source(name: str, config: DataConfiguration) -> Declared:
    """A node providing data of `config`."""
    return Declared(name, outputs=(config,))


def sink(name: str, config: DataConfiguration) -> Declared:
    """A node consuming data of `config`."""
    return Declared(name, inputs=(config,))


def shape_in(graph: Graph, port) -> tuple:
    """The shape of the matched configuration of `port` in `graph`."""
    return port.get_data_configuration(graph).shape
