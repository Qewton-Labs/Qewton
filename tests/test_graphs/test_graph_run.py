import inspect
import pytest
import torch

from qewton.graphs.graphs import Graph
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import BatchAxes, FeatureAxes
from qewton.config.variables import Variable
from qewton.algorithms.building_blocks import Add, Multiply, Square
from qewton.data.datasets import ArrayLikeDataSet
from qewton.data.dataloaders.data_loader import DataLoader


def test_graph_backward_building():
    input_data = torch.zeros((10, 5))

    F = Variable("f", 5)

    input_config = DataConfiguration(BatchAxes(len(input_data)), FeatureAxes(F))
    dataset = ArrayLikeDataSet(data=[input_data], data_configs=[input_config])
    data_loader_1 = DataLoader(
        data_set=dataset,
        batch_size=10,
        splitting_ratio=(1.0, 0.0, 0.0),
        shuffle_data=False,
    )
    data_loader_2 = DataLoader(
        data_set=dataset,
        batch_size=10,
        splitting_ratio=(1.0, 0.0, 0.0),
        shuffle_data=False,
    )
    data_loader_3 = DataLoader(
        data_set=dataset,
        batch_size=10,
        splitting_ratio=(1.0, 0.0, 0.0),
        shuffle_data=False,
    )
    add_node = Add()
    multiply_node = Multiply()
    square_node = Square()
    computation_graph = Graph()
    computation_graph.connect(data_loader_1, add_node.input_ports[0])
    computation_graph.connect(data_loader_2, add_node.input_ports[1])
    computation_graph.connect(add_node, multiply_node.input_ports[0])
    computation_graph.connect(data_loader_2, multiply_node.input_ports[1])
    computation_graph.connect(data_loader_3, square_node.input_ports[0])
    computation_graph.setup()

    nodes_to_run = computation_graph._build_path_to_node(square_node)
    assert data_loader_3 in nodes_to_run
    assert len(nodes_to_run) == 1

    nodes_to_run = computation_graph._build_path_to_node(multiply_node)
    assert data_loader_1 in nodes_to_run
    assert data_loader_2 in nodes_to_run
    assert add_node in nodes_to_run
    assert len(nodes_to_run) == 3

    nodes_to_run = computation_graph._build_path_to_node(add_node)
    assert data_loader_1 in nodes_to_run
    assert data_loader_2 in nodes_to_run
    assert len(nodes_to_run) == 2


def _batch_norms(node) -> list:
    found = []
    for inner in node.inner_graph.nodes:
        if type(inner).__name__ == "FunctionalBatchNorm":
            found.append(inner)
        elif hasattr(inner, "inner_graph"):
            found += _batch_norms(inner)
    return found


def test_a_graphnode_called_on_its_own_runs_in_its_mode():
    import torch

    from qewton.algorithms.dl_models.convolutions.cnn import UNet
    from qewton.optim.base import EvaluationPhase

    model = UNet(in_channels=1, channels=(2, 4), out_channels=1, conv_kernel_size=(3, 3))
    model.setup()
    batch_norms = _batch_norms(model)
    model(torch.rand(1, 1, 8, 8))
    assert batch_norms and not any(bn.training for bn in batch_norms)
    model.set_mode(EvaluationPhase.TRAIN)
    model(torch.rand(1, 1, 8, 8))
    assert all(bn.training for bn in batch_norms)
