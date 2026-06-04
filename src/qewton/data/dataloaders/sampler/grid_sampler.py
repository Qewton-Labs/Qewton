import numpy as np

from ....config.backend import DEFAULT_DL_BACKEND, Backend
from ....config.variables import Variable
from ....geometries.base import Geometry
from ....graphs.nodes import NodeState
from ....optim.parameters.categorical_hyperparameter import CategoricalHyperparameter
from ....optim.parameters.number_hyperparameter import DiscreteHyperparameter

from .point_sampler import PointSampler


class GridSampler(PointSampler):
    """Samples points in a grid-like fashion from a geometry."""

    def __init__(
        self,
        geometry: Geometry,
        n_points: int | DiscreteHyperparameter | CategoricalHyperparameter,
        compute_normals: bool = False,
        normal_name: str | Variable = "normals",
        name: str = "PointSampler",
        state: NodeState = NodeState.FIXED,
        backend: type[Backend] | None = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(
            geometry, n_points, compute_normals, normal_name, name, state, backend
        )
        self.is_static = True
        self.point_cache: np.ndarray | None = None
        self.normal_cache: np.ndarray | None = None

    def sample_points(self):
        if self.point_cache is not None:
            return self.point_cache, self.normal_cache
        if self.is_boundary_geometry:
            self.point_cache, self.normal_cache = self.geometry.sample_grid(
                self.batch_size, include_normals=self.compute_normals  # type: ignore
            )
        else:
            self.point_cache, self.normal_cache = (
                self.geometry.sample_grid(self.batch_size),
                None,
            )
        return self.point_cache, self.normal_cache
