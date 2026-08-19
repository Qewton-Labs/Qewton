import math

from qewton.backends import DEFAULT_DL_BACKEND, TensorType
from qewton.backends.base import ComputingBackend
from qewton.config.variables import Variable
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.geometries.discrete.mesh_stats import mean_edge_length


class VolumeGridGeometry(GridGeometry[TensorType]):
    """A (N1, N2, N3) structured grid of points covering a volumetric mesh's
    bounding box - the solid-box counterpart to PlaneSliceGeometry, for
    resampling a mesh field onto a regular 3D grid (visualization plan,
    roadmap item 5) instead of a stack of 2D plane slices.

    A special case of GridGeometry, exactly like PlaneSliceGeometry: the grid
    points are computed from a mesh's bounding box instead of given directly,
    and `point_filter` marks points outside the mesh via
    `mesh_geometry.contains()`. Feed a VolumeGridGeometry and a
    MeshInterpolationNode's output (e.g. a 3-component vector field) into a
    grid-based vector plot to draw arrows on the resampled field.

    All tensor math runs through mesh_geometry.backend - no numpy dependency.
    Everything computed here is a structural constant (grid geometry, not the
    field), so nothing in this class needs to be differentiable.

    Args:
        mesh_geometry: The volumetric (3D) mesh whose bounding box the grid
            should cover.
        grid_variable (Variable): Composite Variable with exactly 3 scalar
            components, one per grid axis, e.g.
            `Variable("i", 1) * Variable("j", 1) * Variable("k", 1)`. Same
            convention as PlaneSliceGeometry.grid_variable - naming and
            identity are the caller's, for whichever axis a control should
            later target.
        resolution: (N1, N2, N3) grid resolution. If None, derived from the
            mesh's mean edge length (same rationale as PlaneSliceGeometry):
            a grid much finer than the mesh would make interpolation's
            per-cell derivative jumps look like a rendering artifact.
        padding: Fraction of the bounding box's extent to pad on each side,
            so the grid fully covers curved boundaries a tight box would
            clip. 0 (default) uses the exact vertex bounding box.
        backend
    """

    def __init__(
        self,
        mesh_geometry: MeshGeometry,
        grid_variable: Variable,
        resolution: tuple[int, int, int] | None = None,
        padding: float = 0.0,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        assert backend == mesh_geometry.backend, (
            f"VolumeGridGeometry and mesh_geometry should use the same backend, "
            f"found {backend} and {mesh_geometry.backend}"
        )
        assert mesh_geometry.variable.dim == 3, (
            "VolumeGridGeometry only supports 3D meshes, got a "
            f"{mesh_geometry.variable.dim}D mesh_geometry."
        )
        self.mesh_geometry = mesh_geometry
        vertices = mesh_geometry.mesh.vertices

        mins = backend.math.min(vertices, axis=0)
        maxs = backend.math.max(vertices, axis=0)
        if padding:
            extent = maxs - mins
            mins = mins - padding * extent
            maxs = maxs + padding * extent
        self.bounds = (mins, maxs)

        self.resolution = resolution or self._default_resolution(
            backend, mesh_geometry.mesh, mins, maxs
        )
        axes = [
            backend.math.linspace(float(mins[i]), float(maxs[i]), self.resolution[i])
            for i in range(3)
        ]
        xx, yy, zz = backend.math.meshgrid(*axes, indexing="ij")
        point_grid = backend.math.stack([xx, yy, zz], axis=-1)  # (n1, n2, n3, 3)

        point_filter = mesh_geometry.contains(
            backend.math.reshape(point_grid, (-1, point_grid.shape[-1]))
        )
        point_filter = backend.math.reshape(point_filter, point_grid.shape[:-1] + (1,))

        super().__init__(
            variable=grid_variable,
            point_grid=point_grid,
            point_filter=point_filter,
            discretization_of=mesh_geometry,
            backend=backend,
        )

    @staticmethod
    def _default_resolution(backend, mesh, mins, maxs) -> tuple[int, int, int]:
        mean_edge = mean_edge_length(backend, mesh)
        extent = maxs - mins
        return tuple(
            max(2, math.ceil(float(extent[i]) / mean_edge)) for i in range(3)
        )
