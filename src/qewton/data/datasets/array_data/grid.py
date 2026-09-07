from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations.base import DataConfiguration
from qewton.config.variables import Variable
from qewton.data.datasets.array_data.base import ArrayLikeDataSet
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.geometries.discrete.index_grid_geometry import IndexGridGeometry
from qewton.backends import resolve_backend


class GridDataSet(ArrayLikeDataSet):
    """A dataset of one or more array-like data items sharing the same
    grid geometry, each with shape [batch, *grid, feature].

    Args:
        data (Any | list[Any]): One or more array-like data items, each of
            shape (batch_size, *grid_shape, feature_dim). All items must
            share the same grid_shape. An item's batch_size may be smaller
            than the others', as long as it equals 1.
        feature_variables (Variable | list[Variable]): One feature
            Variable per data item, in the same order as data.
        geometry_variable (Variable, optional): The variable connected to
            the grid geometry. Defaults to an auto-named
            Variable("grid", ...) matching the grid's dimensionality.
        point_grid (TensorType, optional): Explicit grid point coordinates,
            of shape grid_shape + (len(grid_shape),). If given, a
            GridGeometry is built from these points instead of an
            IndexGridGeometry. Defaults to None.

    Raises:
        AssertionError: If the data items do not share the same grid
            shape or a compatible batch size, if the number of data items
            does not match the number of feature variables, or if
            point_grid's shape does not match the data's grid shape.
    """

    def __init__(
        self,
        data,
        feature_variables: Variable | list[Variable],
        geometry_variable: Variable = None,
        point_grid=None,
    ):
        data = data if isinstance(data, (list, tuple)) else [data]
        feature_variables = (
            feature_variables
            if isinstance(feature_variables, (list, tuple))
            else [feature_variables]
        )
        backend = resolve_backend(data[0])

        # check that all data items have the same grid and batch shape
        grid_shape = backend.math.shape(data[0])[1:-1]
        assert all(
            backend.math.shape(data[i])[1:-1] == grid_shape for i in range(len(data))
        ), "All data items must have the same grid shape."

        batch_sizes = [backend.math.shape(d)[0] for d in data]
        batch_size = max(batch_sizes)
        for d in data:
            assert (
                backend.math.shape(d)[0] == batch_size or backend.math.shape(d)[0] == 1
            ), "All data items must have the same batch size."

        if geometry_variable is None:
            geometry_variable = Variable("grid", len(grid_shape))

        # if point_grid is provided, we use it to build the geometry
        if point_grid is not None:
            assert all(
                p == g for p, g in zip(backend.math.shape(point_grid), grid_shape)
            ), "The point grid shape has to match the data."
            geometry = GridGeometry(
                geometry_variable, point_grid=point_grid, backend=backend
            )
        else:
            geometry = IndexGridGeometry(geometry_variable, grid_shape, backend=backend)

        geometry_axes = GeometryAxes(geometry)

        configs = []
        assert len(data) == len(
            feature_variables
        ), "The number of data items and feature variables must match."
        for b, v in zip(batch_sizes, feature_variables):
            configs.append(DataConfiguration(BatchAxes(b), geometry_axes, FeatureAxes(v)))

        super().__init__(data, configs)
