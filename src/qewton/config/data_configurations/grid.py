from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.variables import Variable

from .base import DataConfiguration


class GridDataConfig(DataConfiguration):
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
