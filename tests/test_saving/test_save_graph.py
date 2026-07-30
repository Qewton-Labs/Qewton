import torch

from qewton.algorithms.building_blocks import ParameterNode
from qewton import save, load
from qewton.algorithms.building_blocks import Add, Multiply, ReLU
from qewton.algorithms.building_blocks import Reshape
from qewton.graphs.graphs import Graph


def test_simple_graph_save_and_load(tmp_path):
    param_node = ParameterNode(shape=(2,), initial_value=torch.tensor([1.0, 2.0]))
    add_node = Add()
    multiply_node = Multiply()
    relu_node = ReLU()
    reshape_node = Reshape(new_shape=(2, 3))
    graph = Graph()
    graph.connect(param_node, add_node.input_ports[1])
    graph.connect(add_node, multiply_node.input_ports[0])
    graph.connect(multiply_node, relu_node)
    graph.connect(relu_node, reshape_node)
    graph.setup()
    save_path = tmp_path / "graph_test"
    save(graph, save_path)
    loaded_graph = load(save_path)
    assert isinstance(loaded_graph, Graph)
    assert len(loaded_graph.nodes) == len(graph.nodes)
    for node in loaded_graph.nodes:
        for old_node in graph.nodes:
            if node.node_id == old_node.node_id:
                assert type(node) == type(old_node)
                break
    assert len(graph.incoming_edges) == len(loaded_graph.incoming_edges)
