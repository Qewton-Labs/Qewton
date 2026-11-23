"""Test matrix of the relations of axis-changing nodes: node × direction.

Every node of the registry is placed between sources and a sink. In the
forward direction its inputs are known and the configuration of its
output is checked; in the backward direction its output (and possibly
some inputs) are known and the configurations of its other inputs are
checked, after a setup completed the matching (structure flows backward
only then). `None` in an expected shape is a size that stays unknown,
`...` a part whose structure stays unknown. Combinations that do not
hold yet are listed in GAPS with their requirement.
"""

from dataclasses import dataclass, field
from typing import Callable

import pytest
from declared_nodes import B, F, G, dc, shape_in, sink, source, unknown

from qewton.algorithms.building_blocks.array_operations import (
    ConcatNode,
    Flatten,
    Narrow,
    Reshape,
    Squeeze,
    Unsqueeze,
)
from qewton.algorithms.building_blocks.conv import MaxPool1D
from qewton.algorithms.building_blocks.math import Add, Mean, Norm, Std, Sum, Transpose
from qewton.graphs.graphs import Graph


@dataclass
class Case:
    """A node with given configurations around it and what follows.

    `inputs` and `output` are the configurations of the neighbors, `None`
    for an unknown one; `expected` maps a port, `"out"` or the index of an
    input, to its expected shape.
    """

    node: Callable
    inputs: list
    output: Callable | None
    expected: dict = field(default_factory=dict)


FORWARD = {
    "squeeze": Case(lambda: Squeeze(dim=1), [lambda: dc(B(4), G(1), F(3))], None, {"out": (4, 3)}),
    "unsqueeze": Case(lambda: Unsqueeze(dim=1), [lambda: dc(B(4), F(3))], None, {"out": (4, 1, 3)}),
    "transpose": Case(lambda: Transpose(perm=[1, 0]), [lambda: dc(B(4), F(3))], None, {"out": (3, 4)}),
    "sum": Case(lambda: Sum(axis=1), [lambda: dc(B(4), G(5), F(3))], None, {"out": (4, 3)}),
    "mean": Case(lambda: Mean(axis=1), [lambda: dc(B(4), G(5), F(3))], None, {"out": (4, 3)}),
    "std": Case(lambda: Std(axis=1), [lambda: dc(B(4), G(5), F(3))], None, {"out": (4, 3)}),
    "sum_keepdims": Case(
        lambda: Sum(axis=1, keepdims=True), [lambda: dc(B(4), G(5), F(3))], None, {"out": (4, 1, 3)}
    ),
    "norm": Case(lambda: Norm(), [lambda: dc(B(4), F(3))], None, {"out": (4, 1)}),
    "flatten": Case(lambda: Flatten(start_dim=1), [lambda: dc(B(4), G(5), F(3))], None, {"out": (4, 15)}),
    "reshape": Case(lambda: Reshape((4, 15)), [lambda: dc(B(4), G(5), F(3))], None, {"out": (4, 15)}),
    "narrow": Case(
        lambda: Narrow(dim=1, start=0, length=2), [lambda: dc(B(4), G(5), F(3))], None, {"out": (4, 2, 3)}
    ),
    "concat": Case(
        lambda: ConcatNode(concat_dim=-1), [lambda: dc(B(4), F(3)), lambda: dc(B(4), F(2))], None, {"out": (4, 5)}
    ),
    "add_broadcast": Case(
        lambda: Add(), [lambda: dc(B(4), F(3)), lambda: dc(B(1), F(3))], None, {"out": (4, 3), 1: (1, 3)}
    ),
    "max_pool": Case(
        lambda: MaxPool1D(kernel_size=2), [lambda: dc(B(4), F(3), G(8))], None, {"out": (4, 3, 4)}
    ),
}

BACKWARD = {
    "squeeze": Case(lambda: Squeeze(dim=1), [None], lambda: dc(B(4), F(3)), {0: (4, 1, 3)}),
    "unsqueeze": Case(lambda: Unsqueeze(dim=1), [None], lambda: dc(B(4), G(1), F(3)), {0: (4, 3)}),
    "transpose": Case(lambda: Transpose(perm=[1, 0]), [None], lambda: dc(F(3), B(4)), {0: (4, 3)}),
    "sum": Case(lambda: Sum(axis=1), [None], lambda: dc(B(4), F(3)), {0: (4, None, 3)}),
    "sum_keepdims": Case(
        lambda: Sum(axis=1, keepdims=True), [None], lambda: dc(B(4), G(1), F(3)), {0: (4, None, 3)}
    ),
    "norm": Case(lambda: Norm(), [None], lambda: dc(B(4), F(1)), {0: (4, None)}),
    "flatten": Case(lambda: Flatten(start_dim=1), [None], lambda: dc(B(4), F(15)), {0: (4, ...)}),
    "narrow": Case(
        lambda: Narrow(dim=1, start=0, length=2), [None], lambda: dc(B(4), G(2), F(3)), {0: (4, None, 3)}
    ),
    "concat": Case(
        lambda: ConcatNode(concat_dim=-1), [lambda: dc(B(4), F(3)), None], lambda: dc(B(4), F(5)), {1: (4, 2)}
    ),
    "max_pool": Case(
        lambda: MaxPool1D(kernel_size=2), [None], lambda: dc(B(4), F(3), G(4)), {0: (4, 3, None)}
    ),
}

GAPS: dict = {}


def _cases():
    for direction, cases in (("forward", FORWARD), ("backward", BACKWARD)):
        for name in cases:
            reason = GAPS.get((direction, name))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            yield pytest.param(direction, name, marks=marks, id=f"{direction}-{name}")


@pytest.mark.parametrize("direction, name", _cases())
def test_relation(direction, name):
    case = (FORWARD if direction == "forward" else BACKWARD)[name]
    node = case.node()
    graph = Graph()
    for i, config in enumerate(case.inputs):
        provided = config() if config is not None else unknown()
        graph.connect(source(f"source_{i}", provided).output_ports[0], node.input_ports[i])
    expected_out = case.output() if case.output is not None else unknown()
    graph.connect(node.output_ports[0], sink("sink", expected_out).input_ports[0])
    graph.setup()  # structure flows backward when a setup completes the matching

    for port, shape in case.expected.items():
        actual_port = node.output_ports[0] if port == "out" else node.input_ports[port]
        assert shape_in(graph, actual_port) == shape


def test_broadcasting_relates_without_merging():
    add = Add()
    graph = Graph()
    graph.connect(source("x", dc(B(4), F(3))).output_ports[0], add.input_ports[0])
    graph.connect(source("y", dc(B(1), F(3))).output_ports[0], add.input_ports[1])
    graph.connect(add.output_ports[0], sink("out", dc(B(), F())).input_ports[0])
    x, y, out = (add.input_ports[0], add.input_ports[1], add.output_ports[0])
    batch = {p: p.get_data_configuration(graph).axes[0] for p in (x, y, out)}
    assert batch[out] is batch[x]
    assert batch[y] is not batch[x]
    assert batch[out].shape[0].size == 4 and batch[y].shape[0].size == 1
