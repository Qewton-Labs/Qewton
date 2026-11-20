"""Test matrix of ellipses in matching: case × order of `connect` calls.

Every case of `ellipsis_cases` is connected in a graph, its connections
in every possible order; each order must give the same, expected
configurations (config M3, G4), or the same error. Cases or orders that do not
hold yet are listed in GAPS with their reason. The matching core is
checked on the same cases in `test_matrix_merge.py`.
"""

from itertools import permutations

import pytest
from declared_nodes import shape_in
from ellipsis_cases import CASES, port

from qewton.config.errors import DataConfigMismatchError
from qewton.graphs.graphs import Graph

# (case, order) -> reason; None as order stands for every order
GAPS: dict = {}


def _cases():
    for name, case in CASES.items():
        for order in permutations(range(len(case.connections))):
            reason = GAPS.get((name, order)) or GAPS.get((name, None))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            label = "-".join(str(i) for i in order)
            yield pytest.param(name, order, marks=marks, id=f"{name}-{label}")


@pytest.mark.parametrize("name, order", _cases())
def test_ellipsis(name, order):
    case = CASES[name]
    nodes = case.nodes()
    graph = Graph()

    def connect_all():
        for i in order:
            from_name, to_name = case.connections[i]
            graph.connect(port(nodes, from_name), port(nodes, to_name))

    if case.error:
        with pytest.raises(DataConfigMismatchError) as error:
            connect_all()
        assert "data" in str(error.value) and "loss" in str(error.value)
        return
    connect_all()
    for port_name, shape in case.expected.items():
        assert shape_in(graph, port(nodes, port_name)) == shape
