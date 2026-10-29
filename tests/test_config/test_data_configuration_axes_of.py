import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.grid_geometry import GridGeometry

T, X = Variable("t", 1), Variable("x", 1) * Variable("y", 1)
U, P = Variable("u", 1), Variable("p", 1)


@pytest.fixture
def config():
    batch = BatchAxes(3)
    geometry = GeometryAxes(GridGeometry.build_unit_grid(X, 4))
    features = FeatureAxes(U * P)
    return DataConfiguration(batch, geometry, features), batch, geometry, features


def test_axes_are_found_by_identity(config):
    data_config, batch, geometry, _ = config
    assert data_config.axes_of(batch) is batch
    assert data_config.axes_of(geometry) is geometry


def test_a_geometry_variable_and_its_components_lie_in_the_geometry_axes(config):
    data_config, _, geometry, _ = config
    assert data_config.axes_of(X) is geometry
    assert data_config.axes_of(X[0]) is geometry


def test_a_feature_lies_in_the_feature_axes(config):
    data_config, _, _, features = config
    assert data_config.axes_of(U) is features
    assert data_config.axes_of(U * P) is features


@pytest.mark.parametrize("reference", [Variable("q", 1), BatchAxes(3)])
def test_an_unknown_reference_raises(config, reference):
    with pytest.raises(ValueError, match="not part of the data: it holds"):
        config[0].axes_of(reference)


def test_a_variable_held_as_data_lies_in_the_feature_axes():
    """The coordinates of a geometry given as data, e.g. the grid of a
    data loader, are read from the data."""
    from qewton.geometries.discrete.grid_geometry import GridGeometry

    X = Variable("x", 1)
    geometry_axes = GeometryAxes(GridGeometry.build_unit_grid(X, 4))
    feature_axes = FeatureAxes(X)
    config = DataConfiguration(geometry_axes, feature_axes)
    assert config.axes_of(X) is feature_axes


def test_a_missing_reference_names_what_the_data_holds():
    T, U = Variable("t", 1), Variable("u", 1)
    config = DataConfiguration(GeometryAxes(shape=(5,)), FeatureAxes(U))
    with pytest.raises(ValueError) as caught:
        config.axes_of(T)
    assert "a geometry without a Variable (5), the features u" in str(caught.value)
