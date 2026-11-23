"""Models of growing nesting, shared by the test matrices of graphs.

`MODELS` maps a model kind to a factory: a plain node, a GraphNode (an
FCN) and a GraphNode around a graph holding an FCN. Every model maps the
Variable `X` (2 components) to `U` (1 component), so it can be fed by a
sampler on `DOMAIN`.
"""

from typing import Annotated

from qewton.algorithms import FCN
from qewton.backends import TensorType
from qewton.config import Variable
from qewton.config.axes import AxesDim, EllipsisAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node

X, U = Variable("x", 2), Variable("u", 1)
DOMAIN = Rectangle(X, [0.0, 0.0], 1.0, 1.0)


class FirstComponent(Node[TensorType]):
    """A plain node keeping all axes but the features, by one shared
    EllipsisAxes in both annotations."""

    batch = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration(batch, FeatureAxes(shape=(AxesDim(2),)))],
    ) -> Annotated[TensorType, DataConfiguration(batch, FeatureAxes(shape=(AxesDim(1),)))]:
        return 2.0 * x[..., :1]


def fcn() -> FCN:
    return FCN(in_neurons=X, hidden_neurons=4, out_neurons=U, n_hidden_layers=1)


def wrapped(inner_model: Node) -> GraphNode:
    """A GraphNode whose inner graph holds only `inner_model`."""
    graph = Graph()
    graph.add_node(inner_model)
    return GraphNode(
        graph, input_ports=inner_model.input_ports, output_ports=inner_model.output_ports
    )


MODELS = {
    "node": FirstComponent,
    "graph_node": fcn,
    "nested_graph_node": lambda: wrapped(fcn()),
}


def sampled(model: Node, n_points: int) -> Graph:
    """A graph feeding `model` with `n_points` sampled points of `DOMAIN`."""
    graph = Graph()
    graph.connect(RandomUniformSampler(DOMAIN, n_points=n_points), model)
    graph.setup()
    return graph
