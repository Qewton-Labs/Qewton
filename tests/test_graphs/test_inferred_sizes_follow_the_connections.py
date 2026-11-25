"""A size inferred from a connection follows the connections: when the
connection that gave it is removed, or a connection fails, the size is open
again and can be inferred anew (SU1, PL3). A node that is built already keeps
it, since its weights depend on it."""

from typing import Annotated, Any

from qewton.algorithms import FCN
from qewton.config.axes import AxesDim, BatchAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.errors import DataConfigMismatchError
from qewton.graphs import Graph, Node, NodeState
from qewton.optim.parameters.hyperparameter_base import HyperParameter

import pytest


def _network(in_neurons, out_neurons=1):
    return FCN(
        in_neurons=in_neurons, hidden_neurons=4, out_neurons=out_neurons, n_hidden_layers=1
    )


def _input_size(graph, node):
    return graph.config_of(node.input_ports[0]).feature_axes.shape[0].size


@pytest.fixture
def connected():
    """A network whose input size was inferred from a source of size 1."""
    source, target = _network(2, out_neurons=1), _network(None)
    graph = Graph()
    graph.connect(source.output_ports[0], target.input_ports[0])
    return graph, source, target


def test_removing_the_edge_opens_the_inferred_size_again(connected):
    graph, _, target = connected
    graph.remove(graph.incoming_edges[target][0])
    assert target.in_neurons.is_open
    assert _input_size(graph, target) is None


def test_removing_the_source_node_opens_the_inferred_size_again(connected):
    graph, source, target = connected
    graph.remove(source)
    assert target.in_neurons.is_open
    assert _input_size(graph, target) is None


def test_after_removing_the_edge_another_size_can_be_inferred(connected):
    graph, _, target = connected
    graph.remove(graph.incoming_edges[target][0])
    other = _network(5, out_neurons=7)
    graph.connect(other.output_ports[0], target.input_ports[0])
    assert target.in_neurons.value == 7


def test_a_size_given_by_the_user_stays(connected):
    graph, source, _ = connected
    graph.remove(source)
    assert source.out_neurons.value == 1


def test_a_built_node_keeps_its_inferred_size(connected):
    graph, _, target = connected
    target.set_state(NodeState.INITIALIZED)  # as after setup: weights are sized
    graph.remove(graph.incoming_edges[target][0])
    assert target.in_neurons.value == 1


def test_sizes_still_connected_are_inferred_again():
    first, second = _network(2, out_neurons=3), _network(None)
    third = _network(None)
    graph = Graph()
    graph.connect(first.output_ports[0], second.input_ports[0])
    graph.connect(first.output_ports[0], third.input_ports[0])
    graph.remove(graph.incoming_edges[third][0])
    assert second.in_neurons.value == 3
    assert third.in_neurons.is_open


class OpenWidth(Node):
    """Feature size inferred, batch size 20: a merge infers the feature
    size before it finds a contradicting batch size."""

    def __init__(self):
        self.width = HyperParameter.inferred("width")
        super().__init__(state=NodeState.UNINITIALIZED)

    def forward(
        self,
        x: Annotated[
            Any,
            lambda self: DataConfiguration(
                FeatureAxes(shape=(AxesDim(self.width),)), BatchAxes(20)
            ),
        ],
    ): ...


class Width3Batch10(Node):
    def forward(
        self,
    ) -> Annotated[Any, DataConfiguration(FeatureAxes(shape=(AxesDim(3),)), BatchAxes(10))]: ...


def test_a_failed_connection_infers_nothing():
    source, target = Width3Batch10(), OpenWidth()
    graph = Graph()
    graph.add_node(target)
    with pytest.raises(DataConfigMismatchError):
        graph.connect(source.output_ports[0], target.input_ports[0])
    assert target.width.is_open
