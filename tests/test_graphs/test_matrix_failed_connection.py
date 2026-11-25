"""A connection that fails leaves the graph as it was.

Matrix over every way of adding an edge (`connect` with ports or with
nodes, `connect_from_outside_of_graph`, `connect_to_outside_of_graph`,
`add_skip_connection`): after a DataConfigMismatchError the nodes, edges
and configurations are those from before, so a valid connection made
afterwards succeeds.
"""

from typing import Annotated, Any

import pytest

from qewton.config.axes import BatchAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.errors import DataConfigMismatchError
from qewton.config.variables import Variable
from qewton.graphs import Graph, Node

X = Variable("x", 1)
Z = Variable("z", 3)


class Source(Node):
    """Batch size open: the first merge binds it before the features fail."""

    def forward(self) -> Annotated[Any, DataConfiguration(BatchAxes(None), FeatureAxes(X))]: ...


class Mismatching(Node):
    """Batch size 20, features that contradict `Source`."""

    def forward(self, inp: Annotated[Any, DataConfiguration(BatchAxes(20), FeatureAxes(Z))]): ...


class Matching(Node):
    """Batch size 30, features that fit `Source`."""

    def forward(self, inp: Annotated[Any, DataConfiguration(BatchAxes(30), FeatureAxes(X))]): ...


def _connect_ports(graph, from_port, to_port):
    graph.connect(from_port, to_port)


def _connect_nodes(graph, from_port, to_port):
    graph.connect(from_port.node, to_port.node)


def _connect_from_outside(graph, from_port, to_port):
    graph.connect_from_outside_of_graph(from_port, to_port)


def _connect_to_outside(graph, from_port, to_port):
    graph.connect_to_outside_of_graph(from_port, to_port)


def _add_skip_connection(graph, from_port, to_port):
    graph.add_skip_connection(from_port, to_port)


CONNECTIONS = [
    pytest.param(_connect_ports, id="connect-ports"),
    pytest.param(_connect_nodes, id="connect-nodes"),
    pytest.param(_connect_from_outside, id="from-outside"),
    pytest.param(_connect_to_outside, id="to-outside"),
    pytest.param(_add_skip_connection, id="skip-connection"),
]


def _snapshot(graph: Graph, ports) -> tuple:
    """What a failed connection must leave unchanged."""
    return (
        set(graph.nodes),
        {node: list(edges) for node, edges in graph.incoming_edges.items()},
        {node: list(edges) for node, edges in graph.outgoing_edges.items()},
        list(graph.edges_from_outside),
        list(graph.edges_to_outside),
        list(graph.skip_connections),
        [str(graph.config_of(port)) for port in ports],
    )


@pytest.fixture
def nodes():
    return Source(), Mismatching(), Matching()


@pytest.mark.parametrize("connection", CONNECTIONS)
def test_a_failed_connection_raises(connection, nodes):
    source, mismatching, _ = nodes
    graph = Graph()
    graph.add_node(source)
    with pytest.raises(DataConfigMismatchError):
        connection(graph, source.output_ports[0], mismatching.input_ports[0])


@pytest.mark.parametrize("connection", CONNECTIONS)
def test_a_failed_connection_leaves_the_graph_unchanged(connection, nodes):
    source, mismatching, _ = nodes
    graph = Graph()
    graph.add_node(source)
    before = _snapshot(graph, [source.output_ports[0]])
    with pytest.raises(DataConfigMismatchError):
        connection(graph, source.output_ports[0], mismatching.input_ports[0])
    assert _snapshot(graph, [source.output_ports[0]]) == before


@pytest.mark.parametrize("connection", CONNECTIONS)
def test_a_valid_connection_after_a_failed_one_succeeds(connection, nodes):
    source, mismatching, matching = nodes
    graph = Graph()
    graph.add_node(source)
    with pytest.raises(DataConfigMismatchError):
        connection(graph, source.output_ports[0], mismatching.input_ports[0])
    connection(graph, source.output_ports[0], matching.input_ports[0])
    assert graph.config_of(matching.input_ports[0]).axes[0].shape[0].size == 30


class TwoOutputs(Node):
    def forward(
        self,
    ) -> tuple[
        Annotated[Any, DataConfiguration(BatchAxes(None), FeatureAxes(X))],
        Annotated[Any, DataConfiguration(BatchAxes(None), FeatureAxes(X))],
    ]: ...


class SecondInputMismatching(Node):
    def forward(
        self,
        first: Annotated[Any, DataConfiguration(BatchAxes(20), FeatureAxes(X))],
        second: Annotated[Any, DataConfiguration(BatchAxes(20), FeatureAxes(Z))],
    ): ...


def test_connecting_nodes_adds_no_edge_when_a_later_port_pair_fails():
    source, target = TwoOutputs(), SecondInputMismatching()
    graph = Graph()
    graph.add_node(source)
    before = _snapshot(graph, source.output_ports)
    with pytest.raises(DataConfigMismatchError):
        graph.connect(source, target)
    assert _snapshot(graph, source.output_ports) == before
