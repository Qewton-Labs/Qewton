"""Cases of ellipses in matching, shared by the test matrices of graphs
and of the matching core.

Every case is a small chain of declared nodes whose configurations contain
ellipses, the connections between their ports, and the expected shapes
("node.port" -> shape) or the expected error.
"""

from dataclasses import dataclass
from typing import Callable

from declared_nodes import B, F, G, Declared, dc, pass_through, sink, source, unknown

from qewton.config.axes import EllipsisAxes
from qewton.config.data_configurations import DataConfiguration


def _features_3_to_2(name: str, keep_batch: bool = False) -> Declared:
    """Keeps every axis but turns 3 features into 2."""
    e = EllipsisAxes()
    if keep_batch:
        b = B()
        return Declared(name, inputs=(dc(b, e, F(3)),), outputs=(dc(b, e, F(2)),))
    return Declared(name, inputs=(dc(e, F(3)),), outputs=(dc(e, F(2)),))


def _join(name: str) -> Declared:
    """Two inputs of the same structure, one output of it."""
    e = EllipsisAxes()
    config = lambda: DataConfiguration(e, F(3))  # noqa: E731
    return Declared(name, inputs=(config(), config()), outputs=(config(),))


@dataclass
class Case:
    """Nodes, their connections, and the expected shapes ("node.port" →
    shape) or the expected error."""

    nodes: Callable[[], dict]
    connections: list[tuple[str, str]]
    expected: dict | None = None
    error: bool = False


CASES = {
    "binds_one_axis": Case(
        lambda: {
            "data": source("data", dc(B(4), G(5), F(3))),
            "model": _features_3_to_2("model"),
            "loss": sink("loss", DataConfiguration(B(), EllipsisAxes(), F())),
        },
        [("data.out_0", "model.in_0"), ("model.out_0", "loss.in_0")],
        {"model.out_0": (4, 5, 2), "loss.in_0": (4, 5, 2)},
    ),
    "binds_several_axes": Case(
        lambda: {
            "data": source("data", dc(B(4), G(5), G(6), F(3))),
            "model": _features_3_to_2("model"),
            "loss": sink("loss", unknown()),
        },
        [("data.out_0", "model.in_0"), ("model.out_0", "loss.in_0")],
        {"model.out_0": (4, 5, 6, 2), "loss.in_0": (4, 5, 6, 2)},
    ),
    "binds_empty": Case(
        lambda: {
            "data": source("data", dc(B(4), F(3))),
            "model": _features_3_to_2("model", keep_batch=True),
            "loss": sink("loss", unknown()),
        },
        [("data.out_0", "model.in_0"), ("model.out_0", "loss.in_0")],
        {"model.out_0": (4, 2), "loss.in_0": (4, 2)},
    ),
    "concrete_source_last": Case(
        lambda: {
            "data": source("data", dc(B(4), F(3))),
            "pass": pass_through("pass"),
            "model": _features_3_to_2("model", keep_batch=True),
            "loss": sink("loss", DataConfiguration(EllipsisAxes(), F(2))),
        },
        [("data.out_0", "pass.in_0"), ("pass.out_0", "model.in_0"), ("model.out_0", "loss.in_0")],
        {"pass.out_0": (4, 3), "model.out_0": (4, 2), "loss.in_0": (4, 2)},
    ),
    "ellipses_on_both_sides": Case(
        lambda: {
            "data": source("data", DataConfiguration(B(4), EllipsisAxes(), F(3))),
            "pass": pass_through("pass"),
            "loss": sink("loss", DataConfiguration(EllipsisAxes(), G(5), F(3))),
        },
        [("data.out_0", "pass.in_0"), ("pass.out_0", "loss.in_0")],
        {"pass.out_0": (4, ..., 5, 3), "loss.in_0": (4, ..., 5, 3)},
    ),
    "resolved_by_a_later_source": Case(
        lambda: {
            "open": source("open", DataConfiguration(B(4), EllipsisAxes(), F(3))),
            "join": _join("join"),
            "data": source("data", dc(B(4), G(5), G(6), F(3))),
            "loss": sink("loss", unknown()),
        },
        [("open.out_0", "join.in_0"), ("data.out_0", "join.in_1"), ("join.out_0", "loss.in_0")],
        {"open.out_0": (4, 5, 6, 3), "join.out_0": (4, 5, 6, 3), "loss.in_0": (4, 5, 6, 3)},
    ),
    "contradiction": Case(
        lambda: {
            "data": source("data", dc(B(4), F(3))),
            "pass": pass_through("pass"),
            "loss": sink("loss", dc(B(5), F(3))),
        },
        [("data.out_0", "pass.in_0"), ("pass.out_0", "loss.in_0")],
        error=True,
    ),
}


def port(nodes: dict, name: str):
    """The port named "node.port" among `nodes`."""
    node_name, port_name = name.split(".")
    node = nodes[node_name]
    return next(p for p in node.input_ports + node.output_ports if p.name == port_name)
