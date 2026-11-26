"""Cost of matching while building a graph (graphs SU3).

A chain of nodes that pass their data on is connected to a source of
known data, in both orders: source first, and source last. The number of
merges of two configurations must grow at most linearly with the
length of the chain; a full pass per connection would make it grow
quadratically.
"""

from unittest import mock

import pytest
from declared_nodes import B, F, dc, pass_through, source

from qewton.config.matching import Matching
from qewton.graphs.graphs import Graph

ORDERS = ("source_first", "source_last")


def _merges(length: int, order: str) -> int:
    nodes = [pass_through(f"pass_{i}") for i in range(length)]
    data = source("data", dc(B(4), F(3)))
    connections = [(nodes[i].output_ports[0], nodes[i + 1].input_ports[0]) for i in range(length - 1)]
    first = (data.output_ports[0], nodes[0].input_ports[0])
    connections = [first] + connections if order == "source_first" else connections + [first]
    graph = Graph()
    original = Matching.merge_configurations
    with mock.patch.object(
        Matching, "merge_configurations", autospec=True, side_effect=original
    ) as merge:
        for from_port, to_port in connections:
            graph.connect(from_port, to_port)
    return merge.call_count


@pytest.mark.parametrize("order", ORDERS)
def test_matching_grows_linearly_with_the_chain(order):
    short, long = _merges(20, order), _merges(40, order)
    assert long <= 2.2 * short
