from qewton.geometries.continuous.base import (
    ContinuousGeometry,
    ContinuousBoundaryGeometry,
)
from qewton.geometries.continuous.domain_operations.sampler_helper import (
    _boundary_grid_with_n,
    _inside_grid_with_n,
    _inside_random_with_n,
    _boundary_random_with_n,
)
from qewton.backends.base import TensorType
from qewton.config.devices import Device, cpu


class CutGeometry(ContinuousGeometry[TensorType]):
    """Implements the logical cut of two geometries. The cut is implemented via
    "on time sampling" and does not have an explicit representation. When points
    are sampled they are filtered using logical operations.

    Args:
        geometry_a (ContinuousGeometry): The first geometry.
        geometry_b (ContinuousGeometry): The second geometry.
        contained (bool): Whether geometry_b is fully contained within geometry_a.
        backend (type[ComputingBackend[TensorType]], optional): What backend the node
            should use for computations, etc. Defaults to the deep learning
            backend (DEFAULT_DL_BACKEND).
    """

    def __init__(
        self,
        geometry_a: ContinuousGeometry,
        geometry_b: ContinuousGeometry,
        contained=False,
    ):
        assert geometry_a.variable == geometry_b.variable
        assert geometry_a.backend == geometry_b.backend, "Backends do not match!"
        self.geometry_a = geometry_a
        self.geometry_b = geometry_b
        self.contained = contained
        super().__init__(variable=geometry_a.variable, backend=geometry_a.backend)

    def contains(self, points):
        in_a = self.geometry_a.contains(points)
        in_b = self.geometry_b.contains(points)
        return self.backend.math.logical_and(in_a, self.backend.math.logical_not(in_b))

    def bounding_box(self):
        return self.geometry_a.bounding_box()

    def sample_random_uniform(self, n_points: int, device: Device = cpu):
        return _inside_random_with_n(
            self.geometry_a, self.geometry_b, n_points, invert=True, device=device
        )

    def sample_grid(self, n_points: int, device: Device = cpu):
        return _inside_grid_with_n(
            self.geometry_a,
            self.geometry_b,
            n_points,
            invert=True,
            device=device,
        )

    def _get_volume(self):
        if not self.contained:
            # warnings.warn("""Exact volume of this cut is not known, will use the
            #     estimate: volume = geometry_a.volume.
            #     If you need the exact volume for sampling,
            #     use geometry.set_volume()""")
            return self.geometry_a.volume()
        volume_a = self.geometry_a.volume()
        volume_b = self.geometry_b.volume()
        return volume_a - volume_b

    def create_boundary(self):
        return CutBoundaryGeometry(self)


class CutBoundaryGeometry(ContinuousBoundaryGeometry):

    def __init__(self, geometry: CutGeometry):
        assert isinstance(geometry, CutGeometry)
        super().__init__(geometry)
        self.geometry: CutGeometry = geometry  # type: ignore

    def contains(self, points):
        in_a = self.geometry.geometry_a.contains(points)
        in_b = self.geometry.geometry_b.contains(points)
        on_a_bound = self.geometry.geometry_a.boundary.contains(points)
        on_b_bound = self.geometry.geometry_b.boundary.contains(points)
        on_a_part = self.backend.math.logical_and(
            on_a_bound, self.backend.math.logical_not(in_b)
        )
        on_b_part = self.backend.math.logical_and(on_b_bound, in_a)
        on_b_part = self.backend.math.logical_and(
            on_b_part, self.backend.math.logical_not(on_a_bound)
        )
        return self.backend.math.logical_or(on_a_part, on_b_part)

    def _get_volume(self):
        # if not self.geometry.contained:
        # warnings.warn("""Exact volume of this boundary is not known,
        #     will use the estimate:
        #     volume = geometry_a.boundary.volume + geometry_b.boundary.volume.
        #     If you need the exact volume for sampling,
        #     use geometry.set_volume().""")
        volume_a = self.geometry.geometry_a.boundary.volume()
        volume_b = self.geometry.geometry_b.boundary.volume()
        return volume_a + volume_b

    def sample_random_uniform(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        points = _boundary_random_with_n(
            self,
            self.geometry.geometry_a,
            self.geometry.geometry_b,
            n_points,
            device=device,
        )
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def sample_grid(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        points = _boundary_grid_with_n(
            self,
            self.geometry.geometry_a,
            self.geometry.geometry_b,
            n_points,
            device=device,
        )
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def normal(self, points, device: Device = cpu):
        a_normals = self.geometry.geometry_a.boundary.normal(points, device=device)
        b_normals = self.geometry.geometry_b.boundary.normal(points, device=device)
        on_a = self.geometry.geometry_a.boundary.contains(points)
        normals = self.backend.math.where(on_a, a_normals, -b_normals)
        return normals
