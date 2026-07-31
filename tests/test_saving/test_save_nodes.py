import torch

from qewton.algorithms.building_blocks import ParameterNode
from qewton import save, load
from qewton.algorithms.building_blocks import Add
from qewton.algorithms.building_blocks import Reshape


def test_save_and_load_add_node(tmp_path):
    add_node = Add()
    save_path = tmp_path / "add_node_test"
    save(add_node, save_path)
    loaded_add_node = load(save_path)
    assert isinstance(loaded_add_node, Add)
    assert loaded_add_node.node_id == add_node.node_id
    assert loaded_add_node.name == add_node.name
    assert loaded_add_node.backend == add_node.backend


def test_save_and_load_reshape_node(tmp_path):
    reshape_node = Reshape(new_shape=(2, 3))
    save_path = tmp_path / "reshape_node_test"
    save(reshape_node, save_path)
    loaded_reshape_node = load(save_path)
    assert isinstance(loaded_reshape_node, Reshape)
    assert loaded_reshape_node.new_shape == (2, 3)
    assert loaded_reshape_node.node_id == reshape_node.node_id
    assert loaded_reshape_node.name == reshape_node.name
    assert loaded_reshape_node.backend == reshape_node.backend


def test_save_and_load_parameter_node(tmp_path):
    param_node = ParameterNode(shape=(2,), initial_value=torch.tensor([1.0, 2.0]))
    save_path = tmp_path / "param_node_test"
    save(param_node, save_path)
    loaded_param_node = load(save_path)
    assert isinstance(loaded_param_node, ParameterNode)
    assert loaded_param_node.initial_value is not None
    assert torch.allclose(loaded_param_node.initial_value, torch.tensor([1.0, 2.0]))


def test_save_and_load_parameter_node_was_setup(tmp_path):
    param_node = ParameterNode(shape=(2,), initial_value=torch.tensor([1.0, 2.0]))
    param_node.setup()
    save_path = tmp_path / "param_node_test"
    save(param_node, save_path)
    loaded_param_node = load(save_path)
    assert isinstance(loaded_param_node, ParameterNode)
    assert loaded_param_node.initial_value is not None
    assert torch.allclose(loaded_param_node.initial_value, torch.tensor([1.0, 2.0]))
    assert torch.allclose(
        loaded_param_node.trainable_parameters.parameters, torch.tensor([1.0, 2.0])
    )
