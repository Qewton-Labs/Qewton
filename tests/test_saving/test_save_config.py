from qewton import load, save
from qewton.backends.base import TensorType
from qewton.config import (
    Axes,
    AxesDim,
    BatchAxes,
    DataConfiguration,
    EllipsisAxes,
    EllipsisDim,
    FeatureAxes,
    GeometryAxes,
    Variable,
)
from qewton.geometries.base import Geometry


def test_save_and_load_variable_leaf(tmp_path):
    variable = Variable(name="u", dim=1)
    save_path = tmp_path / "variable_leaf_test"

    save(variable, save_path)
    loaded_variable = load(save_path)

    assert isinstance(loaded_variable, Variable)
    assert loaded_variable.name == "u"
    assert loaded_variable.dim == 1
    assert loaded_variable.children == []
    assert loaded_variable.parent is None


def test_save_and_load_variable_tree(tmp_path):
    variable = Variable(
        name="u",
        children=[Variable(name="x", dim=2), Variable(name="t", dim=3)],
    )
    save_path = tmp_path / "variable_tree_test"

    save(variable, save_path)
    loaded_variable = load(save_path)

    assert isinstance(loaded_variable, Variable)
    assert loaded_variable.name == "u"
    assert loaded_variable.dim == 5
    assert len(loaded_variable.children) == 2
    assert loaded_variable.children[0].dim == 2
    assert loaded_variable.children[1].dim == 3
    assert loaded_variable.children[0].parent is loaded_variable
    assert loaded_variable.children[1].parent is loaded_variable


def test_save_and_load_axes_with_ellipsis(tmp_path):
    axes = Axes(2, Ellipsis, AxesDim(4, broadcastable=False))
    save_path = tmp_path / "axes_with_ellipsis_test"

    save(axes, save_path)
    loaded_axes = load(save_path)

    assert isinstance(loaded_axes, Axes)
    assert len(loaded_axes.shape) == 3
    assert loaded_axes.shape[0].size == 2
    assert isinstance(loaded_axes.shape[1], EllipsisDim)
    assert loaded_axes.shape[2].size == 4
    assert loaded_axes.shape[2].broadcastable is False


def test_save_and_load_batch_axes(tmp_path):
    batch_axes = BatchAxes(AxesDim(32, broadcastable=False))
    save_path = tmp_path / "batch_axes_test"

    save(batch_axes, save_path)
    loaded_batch_axes = load(save_path)

    assert isinstance(loaded_batch_axes, BatchAxes)
    assert len(loaded_batch_axes.shape) == 1
    assert loaded_batch_axes.shape[0].size == 32
    assert loaded_batch_axes.shape[0].broadcastable is False


def test_save_and_load_geometry_axes(tmp_path):
    geometry = Geometry(
        variable=Variable(
            name="domain",
            children=[Variable(name="x", dim=2), Variable(name="y", dim=3)],
        ),
        shape=(16, 8),
    )
    geometry_axes = GeometryAxes(geometry=geometry)
    save_path = tmp_path / "geometry_axes_test"

    save(geometry_axes, save_path)
    loaded_geometry_axes = load(save_path)

    assert isinstance(loaded_geometry_axes, GeometryAxes)
    assert loaded_geometry_axes.shape[0].size == 16
    assert loaded_geometry_axes.shape[1].size == 8
    assert loaded_geometry_axes.geometry.shape == (16, 8)
    assert loaded_geometry_axes.geometry.variable.name == "domain"
    assert loaded_geometry_axes.geometry.variable.dim == 5
    assert loaded_geometry_axes.geometry.variable.children[0].dim == 2
    assert loaded_geometry_axes.geometry.variable.children[1].dim == 3
    assert (
        loaded_geometry_axes.geometry.variable.children[0].parent
        is loaded_geometry_axes.geometry.variable
    )


def test_save_and_load_feature_axes(tmp_path):
    feature_axes = FeatureAxes(
        variable=Variable(
            name="u",
            children=[Variable(name="x", dim=2), Variable(name="y", dim=1)],
        )
    )
    save_path = tmp_path / "feature_axes_test"

    save(feature_axes, save_path)
    loaded_feature_axes = load(save_path)

    assert isinstance(loaded_feature_axes, FeatureAxes)
    assert loaded_feature_axes.variables.name == "u"
    assert loaded_feature_axes.variables.dim == 3
    assert loaded_feature_axes.variables.children[0].dim == 2
    assert loaded_feature_axes.variables.children[1].dim == 1
    assert (
        loaded_feature_axes.variables.children[0].parent is loaded_feature_axes.variables
    )


def test_save_and_load_empty_data_configuration(tmp_path):
    data_config = DataConfiguration.empty()
    save_path = tmp_path / "empty_data_config_test"

    save(data_config, save_path)
    loaded_data_config = load(save_path)

    assert isinstance(loaded_data_config, DataConfiguration)
    assert len(loaded_data_config.axes) == 1
    assert isinstance(loaded_data_config.axes[0], EllipsisAxes)
    assert loaded_data_config.dtype is None


def test_save_and_load_data_configuration_with_feature_axes(tmp_path):
    feature_axes = FeatureAxes(
        variable=Variable(
            name="u",
            children=[Variable(name="x", dim=2), Variable(name="y", dim=1)],
        )
    )
    data_config = DataConfiguration(Axes(4), feature_axes, dtype=TensorType)
    save_path = tmp_path / "data_config_with_feature_axes_test"

    save(data_config, save_path)
    loaded_data_config = load(save_path)

    assert isinstance(loaded_data_config, DataConfiguration)
    assert len(loaded_data_config.axes) == 2
    assert loaded_data_config.axes[0].shape[0].size == 4
    assert isinstance(loaded_data_config.feature_axes, FeatureAxes)
    assert loaded_data_config.variable_name == "u"
    assert loaded_data_config.feature_axes.variables.children[0].dim == 2
    assert loaded_data_config.feature_axes.variables.children[1].dim == 1
    assert loaded_data_config.dtype is TensorType


def test_save_and_load_data_configuration_with_batch_geometry_feature_axes(tmp_path):
    geometry = Geometry(
        variable=Variable(name="domain", children=[Variable(name="x", dim=2)]),
        shape=(24, 12),
    )
    feature_axes = FeatureAxes(
        variable=Variable(
            name="u",
            children=[Variable(name="a", dim=4), Variable(name="b", dim=1)],
        )
    )
    data_config = DataConfiguration(
        BatchAxes(AxesDim(8, broadcastable=False)),
        GeometryAxes(geometry=geometry),
        feature_axes,
        dtype=TensorType,
    )
    save_path = tmp_path / "complex_data_config_test"

    save(data_config, save_path)
    loaded_data_config = load(save_path)

    assert isinstance(loaded_data_config, DataConfiguration)
    assert len(loaded_data_config.axes) == 3
    assert isinstance(loaded_data_config.axes[0], BatchAxes)
    assert loaded_data_config.axes[0].shape[0].size == 8
    assert loaded_data_config.axes[0].shape[0].broadcastable is False
    assert isinstance(loaded_data_config.axes[1], GeometryAxes)
    assert loaded_data_config.axes[1].geometry.shape == (24, 12)
    assert loaded_data_config.axes[1].geometry.variable.name == "domain"
    assert loaded_data_config.axes[1].geometry.variable.children[0].dim == 2
    feature_axes_loaded = loaded_data_config.feature_axes
    assert isinstance(feature_axes_loaded, FeatureAxes)
    assert loaded_data_config.variable_name == "u"
    assert feature_axes_loaded.variables.children[0].dim == 4
    assert feature_axes_loaded.variables.children[1].dim == 1
    assert loaded_data_config.dtype is TensorType


def test_save_and_load_data_configuration_with_ellipsis_and_multiple_axes(tmp_path):
    geometry = Geometry(shape=(10, 20))
    feature_axes = FeatureAxes(
        variable=Variable(
            name="state",
            children=[Variable(name="rho", dim=1), Variable(name="v", dim=3)],
        )
    )
    data_config = DataConfiguration(
        BatchAxes(AxesDim(None)),
        EllipsisAxes(),
        GeometryAxes(geometry=geometry),
        feature_axes,
    )
    save_path = tmp_path / "ellipsis_data_config_test"

    save(data_config, save_path)
    loaded_data_config = load(save_path)

    assert isinstance(loaded_data_config, DataConfiguration)
    assert len(loaded_data_config.axes) == 4
    assert isinstance(loaded_data_config.axes[0], BatchAxes)
    assert loaded_data_config.axes[0].shape[0].size is None
    assert isinstance(loaded_data_config.axes[1], EllipsisAxes)
    assert isinstance(loaded_data_config.axes[2], GeometryAxes)
    assert loaded_data_config.axes[2].geometry.shape == (10, 20)
    assert isinstance(loaded_data_config.axes[3], FeatureAxes)
    assert loaded_data_config.variable_name == "state"
    feature_axes_loaded = loaded_data_config.feature_axes
    assert isinstance(feature_axes_loaded, FeatureAxes)
    assert feature_axes_loaded.variables.children[0].dim == 1
    assert feature_axes_loaded.variables.children[1].dim == 3
