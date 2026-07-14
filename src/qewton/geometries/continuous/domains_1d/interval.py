from __future__ import annotations

# import numpy as np

from qewton.geometries.continuous.base import (
    ContinuousGeometry,
    ContinuousBoundaryGeometry,
)
from qewton.geometries.discrete.mesh_geometry import MeshGeometry, Mesh
from qewton.config.variables import Variable
from qewton.config.devices import Device, cpu
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND


class Interval(ContinuousGeometry[TensorType]):
    """Interval class representing a 1D interval domain defined by a lower and
    upper bound.

    Args:
        variable (Variable): The variable associated with the interval, must be 1D.
        lower_bound (float): The lower bound of the interval.
        upper_bound (float): The upper bound of the interval.
        backend (type[ComputingBackend[TensorType]], optional): What backend the node
            should use for computations, etc. Defaults to the deep learning
            backend (DEFAULT_DL_BACKEND).
    """

    def __init__(
        self,
        variable: Variable,
        lower_bound,
        upper_bound,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        assert variable.dim == 1
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__(variable=variable, backend=backend)

    def contains(self, points):
        bigger_then_low = points >= self.lower_bound
        smaller_then_up = points <= self.upper_bound
        return self.backend.math.logical_and(bigger_then_low, smaller_then_up).reshape(
            -1, 1
        )

    def create_mesh(self, max_vertex_distance: float | None = None, device: Device = cpu):
        size = self.upper_bound - self.lower_bound
        if max_vertex_distance is None:
            max_vertex_distance = size
        steps = max(
            1,
            self.backend.math.ceil(self.backend.build_tensor(size / max_vertex_distance)),
        )
        vertices = self.backend.math.linspace(
            self.lower_bound,
            self.upper_bound,
            1 + int(steps),
        )
        vertices = self.backend.math.unsqueeze(vertices, -1)
        cells = [[i, i + 1] for i in range(len(vertices) - 1)]
        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(vertices=vertices, cells=cells),
            discretization_of=self,
            backend=self.backend,
        )

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        points = self.backend.random.uniform((n_points, 1), device=device)
        points *= self.upper_bound - self.lower_bound
        points += self.lower_bound
        return points

    def sample_grid(self, n_points: int, device: Device | str = cpu) -> TensorType:
        points = self.backend.math.linspace(
            self.lower_bound, self.upper_bound, n_points + 2, device=device
        )[1:-1, None]
        return points

    def bounding_box(self):
        return self.backend.build_tensor([self.lower_bound, self.upper_bound])

    def _get_volume(self):
        return self.upper_bound - self.lower_bound

    def create_boundary(self):
        return IntervalBoundary(self)

    @property
    def boundary_left(self) -> IntervalSingleBoundaryPoint:
        """Returns only the left boundary value, useful for the definition
        of initial conditions.
        """
        return IntervalSingleBoundaryPoint(self, side=self.lower_bound)

    @property
    def boundary_right(self) -> IntervalSingleBoundaryPoint:
        """Returns only the right boundary value, useful for the definition
        of end conditions. 
        """
        return IntervalSingleBoundaryPoint(self, side=self.upper_bound, normal_vec=1)


class IntervalBoundary(ContinuousBoundaryGeometry[TensorType]):

    def __init__(self, geometry):
        assert isinstance(geometry, Interval)
        super().__init__(geometry)
        self.geometry: Interval = geometry  # type: ignore

    def contains(self, points):
        close_left = self.backend.math.isclose(
            points, self.backend.build_tensor(self.geometry.lower_bound)
        )
        close_right = self.backend.math.isclose(
            points, self.backend.build_tensor(self.geometry.upper_bound)
        )
        return self.backend.math.logical_or(close_left, close_right).reshape(-1, 1)

    def create_mesh(
        self,
        max_vertex_distance: float | None = None,  # pylint: disable=unused-argument
        device: Device = cpu,  # pylint: disable=unused-argument
    ):
        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(
                vertices=[[self.geometry.lower_bound], [self.geometry.upper_bound]],
                cells=[],
            ),
            discretization_of=self,
            backend=self.backend,
        )

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu, include_normals: bool = False
    ):
        rand_side = self.backend.random.uniform((n_points, 1), device=device)
        random_boundary_index = rand_side < 0.5
        points = self.backend.math.where(
            random_boundary_index, self.geometry.lower_bound, self.geometry.upper_bound
        )
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def sample_grid(
        self, n_points: int, device: Device | str = cpu, include_normals: bool = False
    ):
        lb = self.geometry.lower_bound
        ub = self.geometry.upper_bound
        b_index = self.backend.math.arange(2, device=device)
        b_index = self.backend.math.tile(b_index, (int(n_points / 2.0) + 1,))
        points = self.backend.math.where(
            self.backend.math.equal(b_index[:n_points], 0), lb, ub
        ).reshape(-1, 1)
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def normal(self, points, device: Device | str = cpu) -> TensorType:
        close_left = self.backend.math.isclose(
            points, self.backend.build_tensor(self.geometry.lower_bound)
        )
        return self.backend.math.where(close_left, -1, 1).reshape(-1, 1)

    def _get_volume(self):
        return 2


class IntervalSingleBoundaryPoint(ContinuousBoundaryGeometry[TensorType]):

    def __init__(self, geometry, side, normal_vec=-1):
        assert isinstance(geometry, Interval)
        super().__init__(geometry)
        self.side = side
        self.normal_vec = normal_vec

    def contains(self, points):
        inside = self.backend.math.isclose(points, self.backend.build_tensor(self.side))
        return inside.reshape(-1, 1)

    def create_mesh(
        self,
        max_vertex_distance: float | None = None,  # pylint: disable=unused-argument
        device: Device = cpu,  # pylint: disable=unused-argument
    ):
        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(
                vertices=[[self.side]],
                cells=[],
            ),
            discretization_of=self,
            backend=self.backend,
        )

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu, include_normals: bool = False
    ):
        points = self.side * self.backend.math.ones((n_points, 1), device=device)
        if include_normals:
            normals = self.normal(points)
            return points, normals
        return points

    def sample_grid(
        self, n_points: int, device: Device | str = cpu, include_normals: bool = False
    ):
        return self.sample_random_uniform(
            n_points=n_points, device=device, include_normals=include_normals
        )

    def normal(self, points, device: Device | str = cpu):
        return self.normal_vec * self.backend.math.ones((len(points), 1), device=device)

    def _get_volume(self):
        return 1
