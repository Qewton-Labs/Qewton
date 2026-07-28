import pytest

from qewton.algorithms.dl_models.deeponet.deeponet_fcn import FCNDeepONet
from qewton.backends.base import DeepLearningBackend
from qewton.config.devices import cpu, cuda, cuda_available
from qewton.config.variables import Variable


def all_subclasses(cls):
    """Recursively get all subclasses of a class."""
    result = []
    for sub_cls in cls.__subclasses__():
        result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(DeepLearningBackend)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


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
    assert model.output_dim == 1
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
    assert model.output_dim == 3
    assert model.intermediate_neurons.current_value == 6


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_fcn_deeponet_eval_single_output(backend, device):
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
    branch_points = backend.math.zeros((5, 3), device=device)
    trunk_points = backend.math.zeros((5, 2), device=device)
    model.to(device=device)

    output = model(branch_points, trunk_points)
    print(output.shape)
    assert output.shape[0] == 5
    if len(output.shape) > 1:
        assert output.shape[1] == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("output_strategy", ["split", "split_branch", "split_trunk"])
def test_fcn_deeponet_eval_multi_output_strategies(backend, device, output_strategy):
    model = FCNDeepONet(
        trunk_input=2,
        branch_input=3,
        output=4,
        trunk_hidden_neurons=12,
        branch_hidden_neurons=12,
        trunk_hidden_layers=1,
        branch_hidden_layers=1,
        intermediate_neurons=5,
        output_strategy=output_strategy,
        backend=backend,
    )
    branch_points = backend.math.zeros((6, 3), device=device)
    trunk_points = backend.math.zeros((6, 2), device=device)
    model.to(device=device)

    output = model(branch_points, trunk_points)

    assert output.shape == (6, 4)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_fcn_deeponet_setup_reinitializes_graph(backend, device):
    model = FCNDeepONet(
        trunk_input=2,
        branch_input=3,
        output=3,
        trunk_hidden_neurons=10,
        branch_hidden_neurons=10,
        trunk_hidden_layers=2,
        branch_hidden_layers=2,
        intermediate_neurons=4,
        output_strategy="split_branch",
        backend=backend,
    )

    original_graph = model._graph
    original_branch_graph = model.branch_net._graph
    original_trunk_graph = model.trunk_net._graph
    original_merge_node = model.merge_node

    model.setup()

    assert model._graph is not original_graph
    assert model.branch_net._graph is not original_branch_graph
    assert model.trunk_net._graph is not original_trunk_graph
    assert model.merge_node is not original_merge_node
    assert model.intermediate_neurons.current_value == 4

    branch_points = backend.math.zeros((4, 3), device=device)
    trunk_points = backend.math.zeros((4, 2), device=device)
    model.to(device=device)
    output = model(branch_points, trunk_points)
    assert output.shape == (4, 3)
