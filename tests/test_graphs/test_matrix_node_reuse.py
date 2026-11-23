"""Test matrix of nodes used in graphs: model kind × use × check.

A model kind is a node of growing nesting (a plain node, a GraphNode, a
GraphNode around a graph of GraphNodes). A use puts the same model
instance into one or several graphs, each fed by a sampler with its own
batch size, as PINN pipelines share one model. Every combination checks
the same: the configuration of the model output in each graph, the
configuration inside the model (in the view the graph evaluates), the
values of alternating runs and the values each graph keeps, and that
setting up another graph keeps the weights. Known gaps are listed in
GAPS with their reason and run as strict expected failures.
"""

import pytest
import torch
from model_kinds import DOMAIN, MODELS, fcn

from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler
from qewton.graphs.control_nodes.graph_node import FromFunctionNode, GraphNode, TrackedNode
from qewton.graphs.graphs import Graph

# Uses: name -> batch size of each graph the model is used in
USES = {
    "one_graph": (7,),
    "two_graphs": (7, 5),
}

CHECKS = (
    "outer_config",
    "inner_input_config",
    "inner_output_config",
    "alternating_runs",
    "graph_values",
    "setup_keeps_weights",
)

# (model, use, check) -> reason; None as use stands for every use
GAPS: dict = {}



def _gap(model, use, check):
    return (
        GAPS.get((model, use, check))
        or GAPS.get((model, None, check))
        or GAPS.get((None, None, check))
    )


def _cases():
    for model in MODELS:
        for use in USES:
            for check in CHECKS:
                reason = _gap(model, use, check)
                marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
                yield pytest.param(model, use, check, marks=marks, id=f"{model}-{use}-{check}")


def _build(model_name, use_name):
    model = MODELS[model_name]()
    graphs = []
    for batch_size in USES[use_name]:
        graph = Graph()
        graph.connect(RandomUniformSampler(DOMAIN, n_points=batch_size), model)
        graph.setup()
        graphs.append((graph, batch_size))
    return model, graphs


def _innermost_config(model, graph, outside_edges):
    """Configuration of the innermost port at the model input or output,
    at the place in `graph` that the path of nested GraphNodes leads to."""
    node, path = model, ()
    while isinstance(node, GraphNode):
        path += (node,)
        edge = getattr(node.inner_graph, outside_edges)[0]
        port = edge.to_port if outside_edges == "edges_from_outside" else edge.from_port
        node = port.node
    return port.get_data_configuration(graph, within=path)


def _weights(model):
    tensors = []
    for group in model.trainable_parameters:
        parameters = group.parameters
        if parameters is None:
            continue
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        tensors.extend(p.detach().clone() for p in parameters)
    return tensors


@pytest.mark.parametrize("model_name, use_name, check", _cases())
def test_node_reuse(model_name, use_name, check):
    model, graphs = _build(model_name, use_name)

    if check == "outer_config":
        for graph, n in graphs:
            assert model.output_ports[0].get_data_configuration(graph).shape == (n, 1)

    elif check in ("inner_input_config", "inner_output_config"):
        if not isinstance(model, GraphNode):
            pytest.skip("no inner graph")
        edges, shape = (
            ("edges_from_outside", 2) if check == "inner_input_config"
            else ("edges_to_outside", 1)
        )
        for graph, n in graphs:
            assert _innermost_config(model, graph, edges).shape == (n, shape)

    elif check == "alternating_runs":
        for graph, n in graphs + graphs:
            graph.run()
            assert tuple(graph.value_of(model.output_ports[0]).shape) == (n, 1)

    elif check == "graph_values":
        for graph, _ in graphs:
            graph.run()
        for graph, n in graphs:
            assert tuple(graph.value_of(model.output_ports[0]).shape) == (n, 1)

    elif check == "setup_keeps_weights":
        before = _weights(model)
        if not before:
            pytest.skip("no weights")
        for graph, _ in graphs:
            graph.setup()
        for old, new in zip(before, _weights(model)):
            assert torch.equal(old, new)


class TestCopiesInOneGraph:
    """Two copies of one model in one graph share the weights; values and
    configurations are kept per place, reached through the copy."""

    @staticmethod
    def _graph():
        model = MODELS["graph_node"]()
        first, second = model.copy(), model.copy()
        graph = Graph()
        graph.connect(RandomUniformSampler(DOMAIN, n_points=7), first)
        graph.connect(RandomUniformSampler(DOMAIN, n_points=5), second)
        graph.setup()
        graph.run()
        return graph, model, first, second

    def test_each_copy_has_its_own_values(self):
        graph, model, first, second = self._graph()
        out = model.output_ports[0]
        assert tuple(graph.value_of(first.output_ports[0]).shape) == (7, 1)
        assert tuple(graph.value_of(second.output_ports[0]).shape) == (5, 1)
        assert tuple(graph.value_of(out, within=(first,)).shape) == (7, 1)
        assert tuple(graph.value_of(out, within=(second,)).shape) == (5, 1)

    def test_each_copy_has_its_own_configurations(self):
        graph, model, first, second = self._graph()
        inp = model.input_ports[0]
        assert inp.get_data_configuration(graph, within=(first,)).shape == (7, 2)
        assert inp.get_data_configuration(graph, within=(second,)).shape == (5, 2)

    def test_each_copy_has_its_own_output_configuration(self):
        graph, model, first, second = self._graph()
        out = model.output_ports[0]
        assert out.get_data_configuration(graph, within=(first,)).shape == (7, 1)
        assert out.get_data_configuration(graph, within=(second,)).shape == (5, 1)

    def test_a_node_at_several_places_needs_within(self):
        graph, model, _, _ = self._graph()
        with pytest.raises(ValueError, match="several places"):
            graph.value_of(model.output_ports[0])

    def test_the_weights_are_shared_and_collected_once(self):
        graph, model, _, _ = self._graph()
        collected = [
            id(t)
            for group in graph.collect_trainable_parameters()
            for t in (group.parameters if isinstance(group.parameters, list) else [group.parameters])
            if t is not None
        ]
        own = [id(t) for t in _weights_by_identity(model)]
        assert len(collected) == len(set(collected))
        assert set(collected) == set(own)


def _weights_by_identity(model):
    tensors = []
    for group in model.trainable_parameters:
        parameters = group.parameters
        if parameters is None:
            continue
        tensors.extend([parameters] if isinstance(parameters, torch.Tensor) else parameters)
    return tensors


class TestTrackedFunctions:
    """Graphs built by tracking a function: FromFunctionNode and TrackedNode,
    outside of graphs and inside, also with a model called twice."""

    @staticmethod
    def _sampled(node, n=7):
        graph = Graph()
        graph.connect(RandomUniformSampler(DOMAIN, n_points=n), node)
        graph.setup()
        graph.run()
        return graph

    def test_from_function_node(self):
        model = fcn()
        node = FromFunctionNode(lambda x: 2.0 * model(x))
        x = torch.rand(4, 2)
        assert torch.allclose(node(x), 2.0 * model(x))
        graph = self._sampled(node)
        assert tuple(graph.value_of(node.output_ports[0]).shape) == (7, 1)
        assert tuple(graph.value_of(model.output_ports[0]).shape) == (7, 1)

    def test_tracked_node(self):
        model = fcn()

        class Doubled(TrackedNode):
            def forward(self, x):
                return 2.0 * model(x)

        node = Doubled()
        node.setup()
        x = torch.rand(4, 2)
        assert torch.allclose(node(x), 2.0 * model(x))
        graph = self._sampled(node)
        assert tuple(graph.value_of(node.output_ports[0]).shape) == (7, 1)

    def test_a_model_called_twice_shares_its_weights(self):
        model = fcn()

        class Twice(TrackedNode):
            def forward(self, x):
                return model(x) + model(2.0 * x)

        node = Twice()
        node.setup()
        x = torch.rand(4, 2)
        assert torch.allclose(node(x), model(x) + model(2.0 * x))
        graph = self._sampled(node)
        places = graph._paths_to(model)
        assert len(places) == 2
        for within in places:
            assert tuple(graph.value_of(model.output_ports[0], within=within).shape) == (7, 1)
