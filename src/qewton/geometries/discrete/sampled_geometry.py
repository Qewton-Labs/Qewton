from qewton.geometries.discrete.mesh import Mesh

from qewton.backends import TensorType
from qewton.config.devices import Device
from qewton.geometries.base import DiscreteGeometry, Geometry


class SampledGeometry(DiscreteGeometry[TensorType]):
    """A point cloud sampled from a geometry. This class is used by
    samplers to reference the generated geometry in DataConfigurations.
    Stores a reference to the original geometry, so the sampler can
    access it e.g. in plots.

    Args:
        geometry (Geometry): The geometry to sample from.
        n_points (int): The number of points to sample.
    """

    def __init__(self, geometry: Geometry, n_points: int):

        self.source_geometry = geometry

        self._current_points = None
        self._current_cells = None
        self._mesh_cache: dict[tuple[float | None, Device | str | None], Mesh] = {}

        super().__init__(
            shape=(n_points,), variable=geometry.variable, backend=geometry.backend
        )

    def __deepcopy__(self, memo):
        # Every node's dynamic DataConfiguration is a deepcopy of its static
        # one (Node.copy_data_config_of_port), so a GeometryAxes wrapping a
        # SampledGeometry would otherwise get silently disconnected from the
        # live object the owning sampler mutates via
        # set_current_discretization() each forward() - a copy stuck
        # reporting whatever state existed when the graph was built (usually
        # nothing yet). This object's whole purpose is to be that shared,
        # sampler-owned "current" state - identical to what a Node's own
        # `self` reference already needs to survive graph copying - so it's
        # exempted from copying instead, same as a live connection or cache
        # normally would be.
        return self

    @property
    def discretization_points(self) -> TensorType:
        return self._current_points  # type: ignore

    @discretization_points.setter
    def discretization_points(self, value: TensorType | None) -> None:
        # DiscreteGeometry.__init__ unconditionally assigns
        # `self.discretization_points = discretization_points` (None here,
        # since SampledGeometry doesn't have real points until a sampler
        # actually runs) - this setter just routes that through the same
        # backing field set_current_discretization() writes to, so the
        # property stays live afterwards instead of freezing at None.
        self._current_points = value

    def set_current_discretization(
        self, points: TensorType, cells: TensorType | None = None
    ):
        self._current_points = points
        self._current_cells = cells

    def to_numpy(self) -> None:
        """Converts the current discretization to plain numpy, in place."""
        if self._current_points is not None:
            self._current_points = self.backend.to_numpy(self._current_points)
        if self._current_cells is not None:
            self._current_cells = self.backend.to_numpy(self._current_cells)

    @property
    def mesh(self) -> Mesh | None:
        """A Mesh built from the current discretization, if it has cell
        connectivity - None for a plain point cloud (random/grid sampling,
        no connectivity). Same field name/shape MeshGeometry.mesh always
        has, just not always present here - plotting dispatch (auto_plot,
        MeshPlot, GeometryPlot) checks `geometry.mesh is not None` rather
        than `isinstance(geometry, MeshGeometry)` precisely so it never
        needs to know SampledGeometry exists."""
        if self._current_cells is None:
            return None
        return Mesh(self._current_points, self._current_cells, backend=self.backend)  # type: ignore

    def visualization_mesh(
        self,
        max_vertex_distance: float | None = None,
        device: Device | str | None = None,
    ) -> Mesh:
        """Mesh of the source geometry at the given resolution and device,
        built once per distinct (resolution, device) pair and reused
        afterwards.

        Caching rather than a single slot keeps repeated plots cheap while
        still honouring a changed resolution/device. Within one run only one
        mesh is ever used, so values and vertex positions cannot mismatch.

        `device=None` (default) falls back to Geometry.create_mesh()'s own
        cpu default - PointSampler.set_mesh_mode() is the one that actually
        resolves a concrete device (its own current one, unless overridden),
        so this only sees None when called directly, outside mesh mode.
        """
        cache_key = (max_vertex_distance, device)
        if cache_key not in self._mesh_cache:
            mesh_geometry = (
                self.source_geometry.create_mesh(max_vertex_distance, device)
                if device is not None
                else self.source_geometry.create_mesh(max_vertex_distance)
            )
            self._mesh_cache[cache_key] = mesh_geometry.mesh
        return self._mesh_cache[cache_key]
