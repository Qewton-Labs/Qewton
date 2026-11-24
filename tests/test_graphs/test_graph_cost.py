"""Cost of nesting: an FCN inside 0, 2 and 4 wrapping GraphNodes.

Running a graph runs its steps once, without running the inner graph of a
GraphNode as a whole (graphs GN1), so the number of graph runs per run is
one at every depth. Building it matches every place once (graphs SU3), so the
number of merges grows at most linearly with the depth.
"""

from unittest import mock

import pytest
from model_kinds import fcn, sampled, wrapped

from qewton.config.matching import Matching
from qewton.graphs.graphs import Graph

DEPTHS = (0, 2, 4)


def _nested(depth):
    model = fcn()
    for _ in range(depth):
        model = wrapped(model)
    return model


def _counting(cls, name):
    return mock.patch.object(cls, name, autospec=True, side_effect=getattr(cls, name))


@pytest.mark.parametrize("depth", DEPTHS)
def test_running_runs_no_inner_graph_as_a_whole(depth):
    graph = sampled(_nested(depth), n_points=7)
    with _counting(Graph, "run") as run:
        graph.run()
    assert run.call_count == 1


def test_building_grows_at_most_linearly_with_the_depth():
    counts = []
    for depth in (1, 2, 4):
        with _counting(Matching, "merge_configurations") as merge:
            sampled(_nested(depth), n_points=7)
        counts.append(merge.call_count)
    assert counts[2] <= 2.5 * counts[1]
