from __future__ import annotations

from qewton.config.variables import Variable
from qewton.geometries.base import Geometry, DiscreteGeometry
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.config.devices import Device, cpu
from qewton.config.dtypes import Bool


class GridGeometry(DiscreteGeometry[TensorType]):
    """A grid of points. The underlying points are always in a mesh
    grid like pattern. To create a bit more complex shapes, a filter
    mask can be used.

    Args:
        variable (Variable): The variable connected to this geometry.
        point_grid (TensorType): The grid of points, should be of the
            shape [N_1, N_2, ..., N_d, d], where d is the dimension of
            the above variable.
        point_filter (TensorType | None, optional): A grid of boolean values with the
            same shape as the point_grid, except the last axis should be always 1.
            A value of True, means the corresponding point is inside the geometry,
            False, that it should not be included. Default is None.
        discretization_of (Geometry | None, optional): The geometry this
            grid is a discretization of. Defaults to None.
        backend (type[ComputingBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        variable: Variable,
        point_grid: TensorType,
        point_filter: TensorType | None = None,
        discretization_of: Geometry | None = None,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        assert (
            len(point_grid.shape) - 1 == variable.dim
        ), f"The point grid should have as many axis as the dimension \
            of the variable. Found axis {point_grid.shape[-1]} which has \
            length {len(point_grid.shape) - 1} and does not fit the \
            variable dimension {variable.dim}."
        assert (
            point_grid.shape[-1] == variable.dim
        ), "Dimension of the points in the grid does not fit variable."
        if point_filter is not None:
            assert (
                point_grid.shape[:-1] == point_filter.shape[:-1]
            ), "Filter and grid do not match."
        super().__init__(
            shape=point_grid.shape[:-1],
            variable=variable,
            dim=variable.dim,
            discretization_points=backend.build_tensor(point_grid),
            backend=backend,
        )
        self.discretization_points: TensorType = self.discretization_points
        self.discretization_of = discretization_of

        if point_filter is None:
            self.point_filter = self.backend.math.ones(
                shape=point_grid.shape[:-1] + (1,), dtype=self.backend.dtypes[Bool]
            )
        else:
            self.point_filter = self.backend.cast_dtype(
                point_filter, dtype=self.backend.dtypes[Bool]
            )

        self.cell_volumes: TensorType | None = None

    def _move_data(self, device: Device | str):
        self.discretization_points = self.backend.to(
            self.discretization_points, device=device
        )
        if self.cell_volumes is not None:
            self.cell_volumes = self.backend.to(self.cell_volumes, device=device)

    @classmethod
    def build_unit_grid(
        cls,
        variable: Variable,
        resolution: int,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> GridGeometry:
        """Builds an equidistant grid in the unit cube fitting the provided
        variable.

        Args:
            variable (Variable): The variable connected to this geometry.
            resolution (int): How many points should be used in each
                direction.
            backend (type[ComputingBackend[TensorType]], optional):
                Defaults to DEFAULT_DL_BACKEND.

        Returns:
            GridGeometry: The grid geometry.
        """
        axis = backend.math.linspace(0, 1, resolution)
        axis_list = [axis] * variable.dim
        grid = backend.math.stack(backend.math.meshgrid(*axis_list), axis=-1)
        return GridGeometry(variable=variable, point_grid=grid, backend=backend)

    def _get_volume(self):
        if self.cell_volumes is None:
            self._compute_cell_volumes()
        return self.backend.math.sum(self.cell_volumes)

    def _compute_cell_volumes(self):
        d = self.discretization_points.shape[-1]

        # Compute edge vectors along each grid axis
        edges = []
        for axis in range(d):
            e = self.backend.math.diff(self.discretization_points, axis=axis)
            # remove the last slice along all other axes so all edges
            # have the same cell indexing shape
            slices = [slice(0, n - 1) for n in self.discretization_points.shape[:-1]]
            slices[axis] = slice(None)
            e = e[tuple(slices)]
            edges.append(e)

        # (..., d, d)
        # Each cell gets a matrix whose columns are the edge vectors
        cell_matrix = self.backend.math.stack(edges, axis=-1)
        # remove the ones that should be filtered:
        slices = [slice(0, n - 1) for n in self.discretization_points.shape[:-1]]
        cell_matrix *= self.point_filter[slices].unsqueeze(-1)
        slices = [slice(1, n) for n in self.discretization_points.shape[:-1]]
        cell_matrix *= self.point_filter[slices].unsqueeze(-1)
        # determinant of every cell matrix
        self.cell_volumes = self.backend.math.abs(self.backend.linalg.det(cell_matrix))

    def bounding_box(self):
        bounding_box = []
        for i in range(self.variable.dim):
            min_val = self.backend.math.min(self.discretization_points[..., i])
            max_val = self.backend.math.max(self.discretization_points[..., i])
            bounding_box.append(min_val)
            bounding_box.append(max_val)
        return self.backend.build_tensor(bounding_box)

    def sample_random_uniform_from_discretization(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        self._move_data(device=device)
        points = self.discretization_points.reshape(
            -1, self.discretization_points.shape[-1]
        )
        valid = self.backend.math.squeeze(self.point_filter, -1).reshape(-1)
        valid_points = points[valid]
        valid_idx = self.backend.math.arange(0, len(valid_points), device=device)
        idx = self.backend.random.choice(valid_idx, shape=(n_points, 1), device=device)
        return valid_points[idx]

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        return self.sample_random_uniform_from_discretization(n_points, device)

    def sample_grid(self, n_points: int, device: Device | str = cpu) -> TensorType:
        return self.sample_grid_from_discretization(n_points=n_points, device=device)

    def sample_grid_from_discretization(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        self._move_data(device=device)
        # TODO: Could be done distance based
        points = self.discretization_points.reshape(
            -1, self.discretization_points.shape[-1]
        )
        point_count = len(points)
        if n_points <= point_count:
            return points[:n_points]

        reps = n_points // point_count
        rem = n_points % point_count

        base = self.backend.math.tile(points, (reps, 1))

        extra_idx = self.backend.random.permutation(point_count, device=device)[:rem]
        extra = points[extra_idx]

        return self.backend.math.vstack([base, extra])

    def contains(self, points):
        self._move_data(device=points.device if hasattr(points, "device") else cpu)
        point_check = self.backend.math.zeros(
            (len(points), 1), dtype=self.backend.dtypes[Bool]
        )
        for i, p in enumerate(points):
            point_check[i] = p in self.discretization_points
        return point_check

    def create_boundary(self):
        raise NotImplementedError("Boundary of a point grid currently not implemented.")
