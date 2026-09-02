from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.variables import Variable

from .base import DataConfiguration


class GridDataConfig(DataConfiguration):
    """A DataConfiguration for data on a regular grid, with axes
    [batch, *grid, feature].

    Args:
        shape (tuple[int, ...]): Full data shape, as
            (batch_size, *grid_shape, feature_dim).
        feature_variable (Variable): The variable describing the feature
            axis. Its dim must equal shape[-1].
        geometry_variable (Variable, optional): The variable connected to
            the grid geometry. Its dim must equal len(shape) - 2. Defaults
            to an auto-named Variable("grid", ...) matching the grid's
            dimensionality.
        dtype (optional): The datatype used in this configuration. Defaults
            to None.

    Raises:
        AssertionError: If feature_variable's dim does not match shape[-1].
    """

    def __init__(self, shape, feature_variable, geometry_variable=None, dtype=None):

        from qewton.geometries.discrete.index_grid_geometry import IndexGridGeometry

        if geometry_variable is None:
            geometry_variable = Variable("grid", len(shape[1:-1]))
        assert (
            shape[-1] == feature_variable.dim
        ), f"Feature variable dimension {feature_variable.dim} does not match \
                data shape {shape[-1]}."
        axes = (
            BatchAxes(shape[0]),
            GeometryAxes(IndexGridGeometry(geometry_variable, shape[1:-1])),
            FeatureAxes(feature_variable),
        )
        super().__init__(*axes, dtype=dtype)


GridDataConfiguration = GridDataConfig
