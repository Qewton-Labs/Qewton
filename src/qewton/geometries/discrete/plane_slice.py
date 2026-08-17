import itertools
import math

from qewton.backends import DEFAULT_DL_BACKEND, TensorType
from qewton.backends.base import ComputingBackend
from qewton.config.variables import Variable
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.geometries.discrete.mesh_geometry import MeshGeometry


class PlaneSliceGeometry(GridGeometry[TensorType]):
    """A (k, N1, N2) structured grid of points covering a volumetric mesh's
    cross-section: a stack of `k` planes perpendicular to `normal`, offset
    from the mesh's centroid along it. Feed a PlaneSliceGeometry and a
    MeshInterpolationNode's output into EmbeddedGridPlot (visualization plan,
    section 12).

    A special case of GridGeometry, not a distinct concept: the grid points
    are computed from a mesh and a plane instead of given directly, and
    `point_filter` marks points outside the mesh (via `mesh_geometry.contains()`)
    rather than being supplied by the caller - everything else (bounding_box(),
    sampling, cell volumes) is inherited unchanged.

    Not slice-specific in general - MeshInterpolationNode accepts any
    GridGeometry as its target, so a regular 3D resampling grid or a
    scattered comparison point set work the same way through a plain
    GridGeometry. This class is the concrete constructor for the plane-stack
    case specifically.

    All tensor math runs through mesh_geometry.backend - no numpy dependency,
    so this works uniformly on GPU/torch inputs too. Everything computed here
    is a structural constant (plane geometry, not the field), so nothing in
    this class needs to be differentiable.

    Args:
        mesh_geometry: The volumetric mesh whose cross-section the grid
            should cover.
        normal: Plane normal (dim,). Normalized internally.
        offsets: Offsets along `normal` from the mesh centroid, one plane per
            entry - shape (k,).
        grid_variable (Variable): Composite Variable with exactly 3 scalar
            components - (offset, then the two in-plane grid indices), e.g.
            `Variable("plane", 1) * Variable("u", 1) * Variable("v", 1)`.
            Naming and identity are entirely the caller's: whichever
            component a SliderSpec/FixedSpec should later target (typically
            the offset dimension), the caller builds it and keeps the
            reference. Validated by GridGeometry itself (point_grid.shape[-1]
            must equal grid_variable.dim).
        resolution: (N1, N2) grid resolution. If None, derived from the
            mesh's mean edge length, so nobody mistakes their own mesh
            resolution for a rendering artifact - linear interpolation inside
            cells is continuous but its derivatives jump at cell boundaries,
            so a grid much finer than the mesh makes those boundaries visible.
    """

    def __init__(
        self,
        mesh_geometry: MeshGeometry,
        normal,
        offsets,
        grid_variable: Variable,
        resolution: tuple[int, int] | None = None,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        assert backend == mesh_geometry.backend, (
            f"PlaneSliceGeometry and mesh_geometry should use the same backend, "
            f"found {backend} and {mesh_geometry.backend}"
        )
        self.mesh_geometry = mesh_geometry
        vertices = mesh_geometry.mesh.vertices

        normal = backend.build_tensor(normal)
        self.normal = normal / backend.linalg.norm(normal)
        self.offsets = backend.build_tensor(offsets)
        self.u_hat, self.v_hat = self._plane_basis(backend, self.normal)
        self.centroid = backend.math.mean(vertices, axis=0)

        u_range, v_range = self._plane_extent(
            backend, vertices, self.centroid, self.u_hat, self.v_hat
        )
        self.resolution = resolution or self._default_resolution(
            backend, mesh_geometry.mesh, u_range, v_range
        )
        n1, n2 = self.resolution

        us = backend.math.linspace(u_range[0], u_range[1], n1)
        vs = backend.math.linspace(v_range[0], v_range[1], n2)
        uu, vv = backend.math.meshgrid(us, vs, indexing="ij")

        point_grid = (
            self.centroid
            + backend.math.reshape(self.offsets, (-1, 1, 1, 1)) * self.normal
            + backend.math.unsqueeze(uu, -1) * self.u_hat
            + backend.math.unsqueeze(vv, -1) * self.v_hat
        )  # (k, N1, N2, dim)

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
    def _plane_basis(backend, normal):
        helper = backend.build_tensor([1.0, 0.0, 0.0])
        if abs(float(backend.math.dot(normal, helper))) > 0.9:
            helper = backend.build_tensor([0.0, 1.0, 0.0])
        u_hat = backend.math.cross(normal, helper)
        u_hat = u_hat / backend.linalg.norm(u_hat)
        v_hat = backend.math.cross(normal, u_hat)
        v_hat = v_hat / backend.linalg.norm(v_hat)
        return u_hat, v_hat

    @staticmethod
    def _plane_extent(backend, vertices, centroid, u_hat, v_hat):
        """(u_range, v_range) covering the mesh's cross-section, so a
        rectangular grid patch covers the whole mesh regardless of the
        plane's orientation - not just an axis-aligned bounding box."""
        local = vertices - centroid
        u_coords = backend.math.sum(local * u_hat, axis=1)
        v_coords = backend.math.sum(local * v_hat, axis=1)
        return (
            (float(backend.math.min(u_coords)), float(backend.math.max(u_coords))),
            (float(backend.math.min(v_coords)), float(backend.math.max(v_coords))),
        )

    @staticmethod
    def _mean_edge_length(backend, mesh) -> float:
        vertices = mesh.vertices
        cells = mesh.cells
        pairs = list(itertools.combinations(range(cells.shape[1]), 2))
        edges = backend.math.concatenate(
            [backend.math.sort(cells[:, [a, b]], axis=1) for a, b in pairs], axis=0
        )
        edges = backend.math.unique(edges, axis=0)
        lengths = backend.linalg.norm(
            vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
        )
        return float(backend.math.mean(lengths))

    @classmethod
    def _default_resolution(cls, backend, mesh, u_range, v_range) -> tuple[int, int]:
        mean_edge = cls._mean_edge_length(backend, mesh)
        n1 = max(2, math.ceil((u_range[1] - u_range[0]) / mean_edge))
        n2 = max(2, math.ceil((v_range[1] - v_range[0]) / mean_edge))
        return n1, n2
