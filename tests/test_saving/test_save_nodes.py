from qewton.algorithms.building_blocks import ParameterNode
from qewton.config.saving.callables import save, load
from qewton.graphs.graphs import Graph
from qewton.algorithms.building_blocks import Add
from qewton.algorithms.building_blocks import Reshape
from qewton.data.data_processing.pca import PCANode, InversePCANode
from qewton.data.data_processing.normalization import (
    StdNormalizationNode,
    InverseStdNormalizationNode,
)
from qewton.backends import DEFAULT_DL_BACKEND, _backend_found


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
    if not _backend_found:
        return
    param_node = ParameterNode(
        shape=(2,), initial_value=DEFAULT_DL_BACKEND.build_tensor([1.0, 2.0])
    )
    save_path = tmp_path / "param_node_test"
    save(param_node, save_path)
    loaded_param_node = load(save_path)
    assert isinstance(loaded_param_node, ParameterNode)
    assert loaded_param_node.initial_value is not None
    assert DEFAULT_DL_BACKEND.math.allclose(
        loaded_param_node.initial_value, DEFAULT_DL_BACKEND.build_tensor([1.0, 2.0])
    )


def test_save_and_load_parameter_node_was_setup(tmp_path):
    param_node = ParameterNode(
        shape=(2,), initial_value=DEFAULT_DL_BACKEND.build_tensor([1.0, 2.0])
    )
    param_node.setup()
    save_path = tmp_path / "param_node_test"
    save(param_node, save_path)
    loaded_param_node = load(save_path)
    assert isinstance(loaded_param_node, ParameterNode)
    assert loaded_param_node.initial_value is not None
    assert DEFAULT_DL_BACKEND.math.allclose(
        loaded_param_node.initial_value, DEFAULT_DL_BACKEND.build_tensor([1.0, 2.0])
    )
    assert DEFAULT_DL_BACKEND.math.allclose(
        loaded_param_node.trainable_parameters.parameters,
        DEFAULT_DL_BACKEND.build_tensor([1.0, 2.0]),
    )


def test_save_and_load_pca_node_without_data_fit(tmp_path):
    pca_node = PCANode(n=2, data_source_node=None, scale=True, name="PCA Node")
    save_path = tmp_path / "pca_node_test"
    save(pca_node, save_path)
    loaded_pca_node = load(save_path)
    assert isinstance(loaded_pca_node, PCANode)
    assert loaded_pca_node.n.current_value == 2
    assert loaded_pca_node.scale.current_value is True
    assert loaded_pca_node.name == "PCA Node"


def test_save_and_load_pca_node(tmp_path):
    if not _backend_found:
        return
    pca_node = PCANode(n=5, data_source_node=None, scale=True, name="PCA Node")
    test_data = DEFAULT_DL_BACKEND.random.uniform((10, 50, 4))
    pca_node.fit([test_data])
    save_path = tmp_path / "pca_node_test"
    save(pca_node, save_path)
    loaded_pca_node = load(save_path)
    assert isinstance(loaded_pca_node, PCANode)
    assert loaded_pca_node.n.current_value == 5
    assert loaded_pca_node.scale.current_value is True
    assert loaded_pca_node.name == "PCA Node"
    assert DEFAULT_DL_BACKEND.math.allclose(
        pca_node(test_data)[0], loaded_pca_node(test_data)[0]
    )


def test_save_and_load_inverse_pca_node(tmp_path):
    if not _backend_found:
        return
    pca_node = PCANode(n=2, data_source_node=None, scale=True, name="PCA Node")
    test_data = DEFAULT_DL_BACKEND.random.uniform((10, 50, 4))
    pca_node.fit([test_data])
    inverse_pca_node = InversePCANode(pca_node=pca_node)
    save_path = tmp_path / "inverse_pca_node_test"
    save(inverse_pca_node, save_path)
    loaded_inverse_pca_node = load(save_path)
    input_data = pca_node(test_data)[0]
    assert isinstance(loaded_inverse_pca_node, InversePCANode)
    assert loaded_inverse_pca_node.data_source_node.n.current_value == 2
    assert loaded_inverse_pca_node.data_source_node.scale.current_value is True
    assert loaded_inverse_pca_node.data_source_node.name == "PCA Node"
    assert DEFAULT_DL_BACKEND.math.allclose(
        inverse_pca_node(input_data), loaded_inverse_pca_node(input_data)
    )


def test_save_and_load_std_node(tmp_path):
    if not _backend_found:
        return
    std_node = StdNormalizationNode(
        eps=1e-5, data_source_node=None, name="Normalization Node"  # type: ignore
    )
    test_data = DEFAULT_DL_BACKEND.random.uniform((10, 50, 4))
    std_node.fit([test_data])
    save_path = tmp_path / "std_node_test"
    save(std_node, save_path)
    loaded_std_node = load(save_path)
    assert isinstance(loaded_std_node, StdNormalizationNode)
    assert loaded_std_node.mean is not None
    assert loaded_std_node.std is not None
    assert loaded_std_node.eps == 1e-5
    assert loaded_std_node.name == "Normalization Node"
    assert DEFAULT_DL_BACKEND.math.allclose(
        std_node(test_data)[0], loaded_std_node(test_data)[0]
    )


def test_save_and_load_inverse_std_node(tmp_path):
    if not _backend_found:
        return
    std_node = StdNormalizationNode(
        eps=1e-5, data_source_node=None, name="Normalization Node"  # type: ignore
    )
    test_data = DEFAULT_DL_BACKEND.random.uniform((10, 50, 4))
    std_node.fit([test_data])
    inverse_std_node = InverseStdNormalizationNode(std_node=std_node)
    inverse_std_node.setup(Graph())
    save_path = tmp_path / "inverse_std_node_test"
    save(inverse_std_node, save_path)
    loaded_inverse_std_node = load(save_path)
    input_data = std_node(test_data)[0]
    assert isinstance(loaded_inverse_std_node, InverseStdNormalizationNode)
    assert DEFAULT_DL_BACKEND.math.allclose(
        inverse_std_node(input_data), loaded_inverse_std_node(input_data)
    )
