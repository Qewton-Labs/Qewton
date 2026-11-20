import pytest

from qewton.algorithms.dl_models.deeponet.deeponet_fcn import FCNDeepONet
from qewton.backends.base import DeepLearningBackend
from qewton.config.devices import cpu, cuda, cuda_available
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node, OutputPort


def all_subclasses(cls):
    """Recursively get all subclasses of a class."""
    result = []
    for sub_cls in cls.__subclasses__():
        result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(DeepLearningBackend)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


class Values(Node):
    """A node providing a fixed value."""

    def __init__(self, value):
        super().__init__(name="values")
        self.value = value
        self._output_ports = [
            OutputPort(DataConfiguration.of_value(value), node=self, name="out")
        ]

    def forward(self):
        return self.value


def connected(model, *values) -> Graph:
    """A graph feeding `model` with `values`, set up: the branch size of the
    model is inferred from them."""
    graph = Graph()
    for value, port in zip(values, model.input_ports):
        graph.connect(Values(value), port)
    graph.setup()
    return graph


def evaluated(model, device, *values):
    """The output of `model` for `values` on `device`, run in a graph."""
    graph = connected(model, *values)
    model.to(device=device)
    graph.run()
    return graph.value_of(model.output_ports[0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_fcn_deeponet_initialization_single_output(backend):
    model = FCNDeepONet(
        trunk_input=2,
        branch_input=3,
        output=1,
        trunk_hidden_neurons=16,
        branch_hidden_neurons=16,
        trunk_hidden_layers=2,
        branch_hidden_layers=2,
        intermediate_neurons=8,
        backend=backend,
    )
    assert model.output_dim.value == 1
    assert model.output_strategy == "split"
    assert model.intermediate_neurons.current_value == 8
    assert len(model.input_ports) == 2
    assert len(model.output_ports) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_fcn_deeponet_initialization_with_variable_output(backend):
    output_var = Variable("u", 3)
    model = FCNDeepONet(
        trunk_input=2,
        branch_input=3,
        output=output_var,
        trunk_hidden_neurons=12,
        branch_hidden_neurons=12,
        trunk_hidden_layers=1,
        branch_hidden_layers=1,
        intermediate_neurons=6,
        output_strategy="split",
        backend=backend,
    )
    assert model.output_dim.value == 3
    assert model.intermediate_neurons.current_value == 6
    assert model.output_ports[0].data_configuration.feature_axes.variables is output_var


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_fcn_deeponet_eval_single_output(backend, device):
    model = FCNDeepONet(
        trunk_input=2,
        branch_input=1,
        output=1,
        trunk_hidden_neurons=16,
        branch_hidden_neurons=16,
        trunk_hidden_layers=2,
        branch_hidden_layers=2,
        intermediate_neurons=8,
        backend=backend,
    )
    branch_points = backend.math.zeros((5, 3, 1), device=device)
    trunk_points = backend.math.zeros((5, 3, 2), device=device)

    output = evaluated(model, device, branch_points, trunk_points)
    assert output.shape[0] == 5
    assert output.shape[1] == 3
    assert output.shape[2] == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("output_strategy", ["split", "split_branch", "split_trunk"])
def test_fcn_deeponet_eval_multi_output_strategies(backend, device, output_strategy):
    model = FCNDeepONet(
        trunk_input=2,
        branch_input=1,
        output=4,
        trunk_hidden_neurons=12,
        branch_hidden_neurons=12,
        trunk_hidden_layers=1,
        branch_hidden_layers=1,
        intermediate_neurons=5,
        output_strategy=output_strategy,
        backend=backend,
    )
    branch_points = backend.math.zeros((6, 3, 1), device=device)
    trunk_points = backend.math.zeros((6, 3, 2), device=device)

    output = evaluated(model, device, branch_points, trunk_points)

    assert output.shape == (6, 3, 4)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_fcn_deeponet_setup_rebuilds_a_reset_model(backend, device):
    model = FCNDeepONet(
        trunk_input=2,
        branch_input=1,
        output=3,
        trunk_hidden_neurons=10,
        branch_hidden_neurons=10,
        trunk_hidden_layers=2,
        branch_hidden_layers=2,
        intermediate_neurons=4,
        output_strategy="split_branch",
        backend=backend,
    )
    branch_points = backend.math.zeros((4, 3, 1), device=device)
    trunk_points = backend.math.zeros((4, 3, 2), device=device)
    graph = connected(model, branch_points, trunk_points)

    original_graph = model._graph
    original_branch_graph = model.branch_net._graph
    original_trunk_graph = model.trunk_net._graph
    original_merge_node = model.merge_node

    model.setup()
    assert model._graph is original_graph, "setup keeps a built model"

    model.reset()
    model.setup()
    graph.setup()  # a node set up by hand: its graph is set up again

    assert model._graph is not original_graph
    assert model.branch_net._graph is not original_branch_graph
    assert model.trunk_net._graph is not original_trunk_graph
    assert model.merge_node is not original_merge_node
    assert model.intermediate_neurons.current_value == 4

    model.to(device=device)
    graph.run()
    assert graph.value_of(model.output_ports[0]).shape == (4, 3, 3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fcn_deeponet_can_not_be_called_before_its_branch_size_is_known(backend):
    model = FCNDeepONet(
        trunk_input=2, branch_input=1, output=1, trunk_hidden_neurons=4,
        branch_hidden_neurons=4, trunk_hidden_layers=1, branch_hidden_layers=1,
        intermediate_neurons=3, backend=backend,
    )
    with pytest.raises(RuntimeError, match="Branch input size"):
        model(backend.math.zeros((2, 3, 1)), backend.math.zeros((2, 3, 2)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("output_strategy", ["split", "split_branch", "split_trunk"])
def test_fcn_deeponet_branch_and_trunk_with_their_own_points(backend, output_strategy):
    model = FCNDeepONet(
        trunk_input=2, branch_input=1, output=3, trunk_hidden_neurons=8,
        branch_hidden_neurons=8, trunk_hidden_layers=1, branch_hidden_layers=1,
        intermediate_neurons=4, output_strategy=output_strategy, backend=backend,
    )
    branch_points = backend.math.zeros((2, 6, 1))  # 6 sensors
    trunk_points = backend.math.zeros((2, 5, 7, 2))  # 5 x 7 points

    output = evaluated(model, cpu, branch_points, trunk_points)

    assert output.shape == (2, 5, 7, 3)
    assert model.branch_size.value == 6
