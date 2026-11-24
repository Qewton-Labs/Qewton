"""Test matrix of identity after matching: model kind × link.

Every model kind of `model_kinds` is fed by a sampler. Each link names two
ports whose batch axes must be the same object after setting up the graph
(config M1, M2): the sampler output and the model input, the model input and its
output, and the model input and the first port inside the model. The axes
the sampler declares are read as they are: the model input reads the very
object. Links
that do not hold yet are listed in GAPS with their reason. Broadcasting
(config N3) is checked in `tests/test_config/test_matrix_relate.py`.
"""

import pytest
from model_kinds import MODELS, sampled

from qewton.graphs.control_nodes.graph_node import GraphNode


def _inner_input(model, graph):
    """The innermost port the model input leads to, and the GraphNodes
    leading to it."""
    node, path = model, ()
    while isinstance(node, GraphNode):
        path += (node,)
        port = node.inner_graph.edges_from_outside[0].to_port
        node = port.node
    return port, path


def _batch_axes(graph, port, within=None):
    config = (
        port.get_data_configuration(graph)
        if within is None
        else port.get_data_configuration(graph, within=within)
    )
    return config.axes[0]


def _sampler_output(graph, model):
    (edge,) = graph.incoming_edges[model]
    return edge.from_port


LINKS = {
    "declared_to_input": lambda graph, model: (
        _sampler_output(graph, model).node.points_axes,
        _batch_axes(graph, model.input_ports[0]),
    ),
    "sampler_to_input": lambda graph, model: (
        _batch_axes(graph, _sampler_output(graph, model)),
        _batch_axes(graph, model.input_ports[0]),
    ),
    "input_to_output": lambda graph, model: (
        _batch_axes(graph, model.input_ports[0]),
        _batch_axes(graph, model.output_ports[0]),
    ),
    "input_to_inside": lambda graph, model: (
        _batch_axes(graph, model.input_ports[0]),
        _batch_axes(graph, *_inner_input(model, graph)),
    ),
}

GAPS: dict = {}


def _cases():
    for model in MODELS:
        for link in LINKS:
            if link == "input_to_inside" and model == "node":
                continue  # a plain node has no inside
            reason = GAPS.get((model, link)) or GAPS.get((None, link))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            yield pytest.param(model, link, marks=marks, id=f"{model}-{link}")


@pytest.mark.parametrize("model_name, link", _cases())
def test_identity(model_name, link):
    model = MODELS[model_name]()
    graph = sampled(model, n_points=7)
    first, second = LINKS[link](graph, model)
    assert first is second
