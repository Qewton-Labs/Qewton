from __future__ import annotations
import math

from qewton.config.variables import Variable
from qewton.geometries.base import Geometry, DiscreteGeometry
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.config.devices import Device, cpu
from qewton.config.dtypes import Bool


class PointCloud(DiscreteGeometry[TensorType]):
    """A discrete collection of points.

    Args:
        variable (Variable): The variable connected to this geometry.
        discretization_points (TensorType): The discrete points contained in this
            point cloud.
        discretization_of (Geometry | None, optional): The geometry this
            mesh is a discretization of. Defaults to None.
        backend (type[ComputingBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        variable: Variable,
        discretization_points: TensorType,
        discretization_of: Geometry | None = None,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(
            shape=discretization_points.shape,
            variable=variable,
            dim=variable.dim,
            discretization_points=backend.build_tensor(discretization_points),
            backend=backend,
        )
        self.discretization_of = discretization_of

    def _move_points(self, device: Device | str):
        self.discretization_points = self.backend.to(
            self.discretization_points, device=device
        )

    def bounding_box(self):
        bounding_box = []
        for i in range(self.variable.dim):
            min_val = self.backend.math.min(self.discretization_points[..., i])
            max_val = self.backend.math.max(self.discretization_points[..., i])
            bounding_box.append(min_val)
            bounding_box.append(max_val)
        return self.backend.build_tensor(bounding_box)

    def _get_volume(self):
        bounding_box = self.bounding_box()
        side_dist = bounding_box[1::2] - bounding_box[::2]
        return math.prod(side_dist)

    def create_boundary(self):
        raise NotImplementedError("Point clouds dont have a boundary.")

    def sample_random_uniform_from_discretization(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        point_count = len(self.discretization_points)
        idx = self.backend.random.choice(
            point_count,
            shape=n_points,
            replace=(n_points > point_count),
            device=device,
        )
        self._move_points(device=device)
        return self.discretization_points[idx]

    def sample_grid_from_discretization(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        self._move_points(device=device)
        # TODO: Could be done distance based
        point_count = len(self.discretization_points)
        if n_points <= point_count:
            return self.discretization_points[:n_points]

        reps = n_points // point_count
        rem = n_points % point_count

        base = self.backend.math.tile(self.discretization_points, (reps, 1))

        extra_idx = self.backend.random.permutation(point_count, device=device)[:rem]
        extra = self.discretization_points[extra_idx]

        return self.backend.math.vstack([base, extra])

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        return self.sample_random_uniform_from_discretization(
            n_points=n_points, device=device
        )

    def sample_grid(self, n_points: int, device: Device | str = cpu) -> TensorType:
        return self.sample_grid_from_discretization(n_points=n_points, device=device)

    def contains(self, points):
        self._move_points(device=points.device if hasattr(points, "device") else cpu)
        point_check = self.backend.math.zeros(
            (len(points), 1), dtype=self.backend.dtypes[Bool]
        )
        for i, p in enumerate(points):
            point_check[i] = p in self.discretization_points
        return point_check
