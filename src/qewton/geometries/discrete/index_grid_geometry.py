from __future__ import annotations

from qewton.backends import DEFAULT_DL_BACKEND
from qewton.backends.base import ComputingBackend, TensorType
from qewton.config.devices import Device
from qewton.config.dtypes import Bool, Float32, Int32
from qewton.config.variables import Variable
from qewton.geometries.base import DiscreteGeometry
from qewton.geometries.discrete.grid_geometry import GridGeometry


class IndexGridGeometry(GridGeometry[TensorType]):
    """A grid whose coordinates are its own indices, materialized on demand.

    For data stored on a grid with no known physical positions - images,
    solution arrays read from disk. Index coordinates are genuine coordinates
    in index space, so the full GridGeometry contract holds: contains(),
    sampling and cell volumes stay meaningful. Volume is then a cell count,
    not a physical quantity.

    Nothing is weakened relative to GridGeometry - the coordinates exist, they
    are simply not stored until something asks. That is the whole point: a
    512^3 grid would otherwise carry a 1.6 GB coordinate array describing
    positions that are implicit anyway.

    Args:
        variable (Variable): The variable connected to this geometry. Its dim
            must equal len(shape). Build it composite
            (Variable("i", 1) * Variable("j", 1)) if a SliderSpec or FacetSpec
            should later target a single grid axis.
        shape (tuple[int, ...]): Grid extent per axis.
        point_filter (TensorType | None): Same meaning as in GridGeometry -
            shape[:-1] + (1,), True for included points.
    """

    def __init__(
        self,
        variable: Variable,
        shape: tuple[int, ...],
        point_filter: TensorType | None = None,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        assert len(shape) == variable.dim, (
            f"Grid has {len(shape)} axes but the variable has dimension "
            f"{variable.dim}; they must match."
        )
        self._points: TensorType | None = None
        self._grid_shape = tuple(shape)

        # GridGeometry.__init__ is deliberately skipped: it requires a
        # materialized point_grid and derives shape and filter from it.
        # Everything it sets up that does not depend on the points is
        # reproduced below - see the notes on keeping this in sync.
        DiscreteGeometry.__init__(
            self,
            shape=self._grid_shape,
            variable=variable,
            dim=variable.dim,
            discretization_points=None,
            backend=backend,
        )

        if point_filter is None:
            self.point_filter = self.backend.math.ones(
                shape=self._grid_shape + (1,), dtype=self.backend.dtypes[Bool]
            )
        else:
            assert (
                tuple(point_filter.shape[:-1]) == self._grid_shape
            ), "Filter and grid shape do not match."
            self.point_filter = self.backend.cast_dtype(
                point_filter, dtype=self.backend.dtypes[Bool]
            )

        self.discretization_of = None  # not a discretization of a continuous domain
        self.cell_volumes: TensorType | None = None
        self._device = None

    # -- lazy coordinates ---------------------------------------------------

    @property
    def discretization_points(self) -> TensorType:
        """Index coordinates, built on first access and cached afterwards."""
        if self._points is None:
            self._points = self._build_index_grid()
        return self._points

    @discretization_points.setter
    def discretization_points(self, value) -> None:
        # DiscreteGeometry.__init__ and _move_data assign here; route those
        # into the lazy slot instead of shadowing the property.
        self._points = value

    @property
    def is_materialized(self) -> bool:
        """Whether the coordinate array currently exists in memory. Useful in
        tests to assert that a code path stayed lazy."""
        return self._points is not None

    def _build_index_grid(self) -> TensorType:
        math = self.backend.math
        axes = [
            self.backend.cast_dtype(math.arange(n), dtype=self.backend.dtypes[Int32])
            for n in self._grid_shape
        ]
        grid = math.stack(math.meshgrid(*axes, indexing="ij"), axis=-1)
        return self.backend.to(grid, device=self._device)

    # -- overrides that avoid materializing ---------------------------------

    def _move_data(self, device: Device | str) -> None:
        # Never materialize just to move - an unbuilt grid is rebuilt on the
        # new device at first access.
        if self._points is not None:
            self._points = self.backend.to(self._points, device=device)
        else:
            self._device = device
        if self.cell_volumes is not None:
            self.cell_volumes = self.backend.to(self.cell_volumes, device=device)

    def bounding_box(self):
        lo = self.backend.math.zeros(shape=(self.dim,))
        hi = self.backend.build_tensor([float(n - 1) for n in self._grid_shape])
        return lo, hi

    def _compute_cell_volumes(self) -> None:
        """Every cell of an index grid is a unit hypercube, so volumes follow
        from the filter alone - no coordinate array and no diff needed."""
        math = self.backend.math
        cell_shape = tuple(n - 1 for n in self._grid_shape)
        volumes = math.ones(shape=cell_shape, dtype=self.backend.dtypes[Float32])
        lower_corner = tuple(slice(0, n - 1) for n in self._grid_shape)
        volumes = volumes * self.backend.cast_dtype(
            self.point_filter[lower_corner][..., 0],
            dtype=self.backend.dtypes[Float32],
        )
        self.cell_volumes = self.backend.to(volumes, device=self._device)
