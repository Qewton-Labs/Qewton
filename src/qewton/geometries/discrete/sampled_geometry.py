"""Geometry holding the points a sampler currently provides."""

from qewton.geometries.discrete.mesh import Mesh

from qewton.backends import TensorType
from qewton.config.devices import Device
from qewton.config.variables import Variable
from qewton.geometries.base import DiscreteGeometry, Geometry, vertex_distance_for
from qewton.geometries.product import ProductGeometry


class SampledGeometry(DiscreteGeometry[TensorType]):
    """Point cloud sampled from a geometry.

    Samplers use it to reference their generated points in data
    configurations. It holds the points (and optionally cells) of the
    current sampling step, set by the owning sampler through
    :meth:`set_current_discretization`, and a reference to the source
    geometry, e.g. for plotting.

    Args:
        geometry: The geometry the points are sampled from.
        n_points: The number of sampled points.

    Attributes:
        source_geometry (Geometry): The geometry the points are sampled from.
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

        self.source_geometry = geometry

        self._current_points = None
        self._current_cells = None
        self._current_factors: list | None = None
        self._current_product = None
        self._is_numpy = False
        self._mesh_cache: dict[tuple, Geometry] = {}

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
    def factors(self) -> list | None:
        return self._current_factors

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

    def visualization_mesh(
        self,
        max_vertex_distance: float | dict | None = None,
        device: Device | str | None = None,
    ) -> Mesh:
        """Return the mesh of the source geometry for visualization.

        Args:
            max_vertex_distance: The mesh resolution, see
                :meth:`visualization_geometry`.
            device: The device of the mesh, see :meth:`visualization_geometry`.

        Returns:
            Mesh: The mesh of :meth:`visualization_geometry`.
        """
        return self.visualization_geometry(max_vertex_distance, device).mesh

    def visualization_geometry(
        self,
        max_vertex_distance: float | dict[Variable, float] | None = None,
        device: Device | str | None = None,
    ) -> Geometry:
        """Return the meshed source geometry for visualization.

        The mesh is built once per distinct ``(max_vertex_distance,
        device)`` pair and cached.

        Args:
            max_vertex_distance: The maximum vertex distance of the mesh. A
                dict gives one value per geometry variable: a
                :class:`~qewton.geometries.product.ProductGeometry` receives it
                as-is to mesh each factor, any other geometry uses the entry of
                its own variable.
            device: The device of the mesh. ``None`` uses the default device of
                :meth:`~qewton.geometries.base.Geometry.create_mesh`.

        Returns:
            Geometry: The meshed source geometry.
        """
        cache_key = (
            frozenset(max_vertex_distance.items())
            if isinstance(max_vertex_distance, dict)
            else max_vertex_distance,
            device,
        )
        if cache_key not in self._mesh_cache:
            if not isinstance(self.source_geometry, ProductGeometry):
                max_vertex_distance = vertex_distance_for(
                    max_vertex_distance, self.source_geometry.variable
                )
            self._mesh_cache[cache_key] = (
                self.source_geometry.create_mesh(max_vertex_distance, device)
                if device is not None
                else self.source_geometry.create_mesh(max_vertex_distance)
            )
        return self._mesh_cache[cache_key]
