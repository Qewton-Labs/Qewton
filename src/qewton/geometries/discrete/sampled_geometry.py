"""Geometry holding the points a sampler currently provides."""

import numpy as np

from qewton.geometries.discrete.mesh import Mesh

from qewton.backends import TensorType
from qewton.geometries.base import DiscreteGeometry, Geometry, Structure


class SampledGeometry(DiscreteGeometry[TensorType]):
    """Point cloud sampled from a geometry.

    Samplers use it to reference their generated points in data
    configurations. It holds the points (and optionally cells) of the
    current sampling step, set by the owning sampler through
    :meth:`set_current_discretization`. Meshing the source geometry for
    plotting is done by the sampler, which owns that geometry.

    Args:
        geometry: The geometry the points are sampled from; its variable and
            backend are taken over.
        n_points: The number of sampled points.

    Attributes:
        discretization_points (TensorType or None): The current points.
        factors (list or None): The meshed factor geometries if the current
            points are the vertices of a product mesh (first factor varying
            slowest), otherwise ``None``.
        mesh (Mesh or None): A :class:`~qewton.geometries.discrete.mesh.Mesh`
            built from the current points and cells, or ``None`` for a plain
            point cloud without cell connectivity. For a product mesh the cells
            are built on first access.

    Notes:
        Deep copies return the same object, so copies of a data configuration
        keep seeing the points the sampler sets.
    """

    def __init__(self, geometry: Geometry, n_points: int):
        self._current_points = None
        self._current_cells = None
        self._current_factors: list | None = None
        self._current_product = None
        self._is_numpy = False

        super().__init__(
            shape=(n_points,), variable=geometry.variable, backend=geometry.backend
        )

    def __deepcopy__(self, memo):
        """Return this object itself instead of a copy."""
        # Shared, sampler-owned state: copies of a DataConfiguration must
        # keep referencing the object the sampler updates.
        return self

    @property
    def discretization_points(self) -> TensorType:
        return self._current_points  # type: ignore

    @discretization_points.setter
    def discretization_points(self, value: TensorType | None) -> None:
        # Uses the same field as set_current_discretization(), including
        # the None assigned by DiscreteGeometry.__init__.
        self._current_points = value

    def set_current_discretization(
        self, points: TensorType, cells: TensorType | None = None, product=None
    ):
        """Store the points (and cells) the sampler currently provides.

        Args:
            points: The current points.
            cells: Cell connectivity of the points.
            product (ProductMeshGeometry or None): Set when the points are the
                vertices of a product mesh. Its factors are exposed as
                `factors`, and its cells are only built when `mesh` is
                accessed.
        """
        self._current_points = points
        self._current_cells = cells
        self._current_product = product
        self._current_factors = product.factors if product is not None else None
        self._is_numpy = False

    @property
    def structure(self) -> Structure:
        return Structure.POINTS if self.mesh is None else Structure.MESH

    @property
    def topological_dim(self) -> int:
        mesh = self.mesh
        return 0 if mesh is None else mesh.cells.shape[1] - 1

    @property
    def cells(self) -> np.ndarray | None:
        mesh = self.mesh
        return None if mesh is None else np.asarray(self.backend.to_numpy(mesh.cells))

    @property
    def factors(self) -> list | None:
        return self._current_factors

    def subproduct(self, factors: list) -> DiscreteGeometry:
        """See :meth:`~qewton.geometries.base.DiscreteGeometry.subproduct`;
        the product of the current product mesh's factors."""
        if self._current_product is None:
            return super().subproduct(factors)
        return self._current_product.subproduct(factors)

    def to_numpy(self) -> None:
        """Convert the current points and cells to numpy arrays in place."""
        if self._current_points is not None:
            self._current_points = self.backend.to_numpy(self._current_points)
        if self._current_cells is not None:
            self._current_cells = self.backend.to_numpy(self._current_cells)
        self._is_numpy = True

    @property
    def mesh(self) -> Mesh | None:
        if self._current_cells is None and self._current_product is not None:
            cells = self._current_product.mesh.cells
            self._current_cells = self.backend.to_numpy(cells) if self._is_numpy else cells
        if self._current_cells is None:
            return None
        return Mesh(self._current_points, self._current_cells, backend=self.backend)  # type: ignore
