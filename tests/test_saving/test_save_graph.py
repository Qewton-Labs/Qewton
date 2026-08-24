from qewton.algorithms.building_blocks import ParameterNode
from qewton import save, load
from qewton.algorithms.building_blocks import Add, Multiply, ReLU
from qewton.algorithms.building_blocks import Reshape
from qewton.algorithms.dl_models.fcn import FCN, DeepRitzNet
from qewton.algorithms.dl_models.cnn import CNN, UNet
from qewton.algorithms.dl_models.harmonic_fcn import HarmonicFCN
from qewton.graphs.graphs import Graph
from qewton.backends import DEFAULT_DL_BACKEND, _backend_found


def test_simple_graph_save_and_load(tmp_path):
    if not _backend_found:
        return
    param_node = ParameterNode(
        shape=(2,), initial_value=DEFAULT_DL_BACKEND.build_tensor([1.0, 2.0])
    )
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


def test_save_and_load_fcn(tmp_path):
    fcn = FCN(in_neurons=2, hidden_neurons=4, out_neurons=1, n_hidden_layers=3)
    save_path = tmp_path / "fcn_test"
    save(fcn, save_path)
    loaded_fcn = load(save_path)
    assert isinstance(loaded_fcn, FCN)
    if _backend_found:
        input_data = DEFAULT_DL_BACKEND.random.uniform((10, 2))
        output_original = fcn(input_data)
        output_loaded = loaded_fcn(input_data)
        assert DEFAULT_DL_BACKEND.math.allclose(output_original, output_loaded)


def test_save_and_load_fcn_more_complex(tmp_path):
    fcn = FCN(
        in_neurons=2, hidden_neurons=25, out_neurons=2, n_hidden_layers=5, bias=False
    )
    save_path = tmp_path / "fcn_test"
    save(fcn, save_path)
    loaded_fcn = load(save_path)
    assert isinstance(loaded_fcn, FCN)
    if _backend_found:
        input_data = DEFAULT_DL_BACKEND.build_tensor([[1.0, 2.0], [3.0, 4.0]])
        output_original = fcn(input_data)
        output_loaded = loaded_fcn(input_data)
        assert DEFAULT_DL_BACKEND.math.allclose(output_original, output_loaded)


def test_save_and_load_deep_ritz_net(tmp_path):
    fcn = DeepRitzNet(in_neurons=2, width=10, out_neurons=1, depth=1)
    save_path = tmp_path / "deepritz_test"
    save(fcn, save_path)
    loaded_fcn = load(save_path)
    assert isinstance(loaded_fcn, DeepRitzNet)
    if _backend_found:
        input_data = DEFAULT_DL_BACKEND.build_tensor([[1.0, 2.0], [3.0, 4.0]])
        output_original = fcn(input_data)
        output_loaded = loaded_fcn(input_data)
        assert DEFAULT_DL_BACKEND.math.allclose(output_original, output_loaded)


def test_save_and_load_cnn(tmp_path):
    save_path = tmp_path / "cnn_test"
    cnn = CNN(
        in_channels=2,
        hidden_channels=16,
        out_channels=1,
        n_hidden_layers=3,
        kernel_size=(3, 3),
    )
    cnn.setup()
    save(cnn, save_path, replace=True)
    cnn_loaded = load(save_path)
    assert isinstance(cnn_loaded, CNN)
    if _backend_found:
        input_data = DEFAULT_DL_BACKEND.random.uniform((1, 2, 5, 5))
        output_original = cnn(input_data)
        output_loaded = cnn_loaded(input_data)
        assert DEFAULT_DL_BACKEND.math.allclose(output_original, output_loaded)


def test_save_and_load_unt(tmp_path):
    save_path = tmp_path / "unet_test"
    unet = UNet(
        in_channels=2,
        channels=(16, 32, 64),
        out_channels=1,
        conv_kernel_size=(3, 3),
    )
    unet.setup()
    save(unet, save_path, replace=True)
    unet_loaded = load(save_path)
    assert isinstance(unet_loaded, UNet)
    if _backend_found:
        input_data = DEFAULT_DL_BACKEND.random.uniform((1, 2, 20, 20))
        output_original = unet(input_data)
        output_loaded = unet_loaded(input_data)
        assert DEFAULT_DL_BACKEND.math.allclose(output_original, output_loaded)


def test_save_and_load_harmonic_fcn(tmp_path):
    save_path = tmp_path / "harmonic_fcn_test"
    harmonic_fcn = HarmonicFCN(
        input_dim=2,
        hidden_neurons=4,
        output_dim=1,
        n_hidden_layers=3,
        max_harmonic=3,
        bias=True,
        activation=ReLU,
    )
    harmonic_fcn.setup()
    save(harmonic_fcn, save_path, replace=True)
    loaded_harmonic_fcn = load(save_path)
    assert isinstance(loaded_harmonic_fcn, HarmonicFCN)
    if _backend_found:
        input_data = DEFAULT_DL_BACKEND.build_tensor([[1.0, 2.0], [3.0, 4.0]])
        output_original = harmonic_fcn(input_data)
        output_loaded = loaded_harmonic_fcn(input_data)
        assert DEFAULT_DL_BACKEND.math.allclose(output_original, output_loaded)
