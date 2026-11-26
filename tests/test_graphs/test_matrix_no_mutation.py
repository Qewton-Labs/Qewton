"""Test matrix of shared objects under matching: model kind × shared object.

A model of every kind of `model_kinds` is set up, then used in two graphs
with different batch sizes (config M4, graphs PL3). Matching in these graphs must leave
untouched what is shared between them: every object of the static
configurations of all ports (also inside GraphNodes), the Variables of the
model and the geometry of the samplers. Combinations that do not hold yet are listed in
GAPS with their reason.
"""

import pytest
from model_kinds import DOMAIN, MODELS, U, X, sampled

from qewton.graphs.control_nodes.graph_node import GraphNode


def _nodes(model):
    """`model` and every node inside it, at any depth."""
    nodes, todo = [], [model]
    while todo:
        node = todo.pop()
        nodes.append(node)
        if isinstance(node, GraphNode):
            todo.extend(node.inner_graph.nodes)
    return nodes


def _state(item) -> tuple:
    """Every attribute of a declared axes or dim and of its parts: values,
    or the identity of objects."""
    attributes = tuple(
        sorted(
            (key, value if isinstance(value, (int, float, str, bool, type(None))) else id(value))
            for key, value in vars(item).items()
        )
    )
    return (id(item), attributes, tuple(_state(part) for part in item.parts))


def _static_configurations(model):
    """For every port: the state of every object of its static
    configuration."""
    return {
        port: (port.data_configuration.dtype, [_state(a) for a in port.data_configuration.axes])
        for node in _nodes(model)
        for port in node.input_ports + node.output_ports
    }


def _variables():
    return [(v, v.name, v.dim, v.shape) for v in (X, U)]


def _geometry():
    return (DOMAIN.variable, DOMAIN.variable.name, DOMAIN.dim)


SHARED = {
    "static_configurations": _static_configurations,
    "variables": lambda model: _variables(),
    "sampler_geometry": lambda model: _geometry(),
}

GAPS = {}


def _cases():
    for model in MODELS:
        for shared in SHARED:
            reason = GAPS.get((model, shared)) or GAPS.get((None, shared))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            yield pytest.param(model, shared, marks=marks, id=f"{model}-{shared}")


@pytest.mark.parametrize("model_name, shared", _cases())
def test_matching_leaves_shared_objects_untouched(model_name, shared):
    model = MODELS[model_name]()
    model.setup()
    snapshot = SHARED[shared]
    before = snapshot(model)
    for n_points in (7, 5):
        sampled(model, n_points).run()
    assert snapshot(model) == before
