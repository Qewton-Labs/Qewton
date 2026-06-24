from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.config.variables import Variable
from qewton.geometries.base import Geometry
from qewton.graphs.nodes import NodeState
from qewton.optim.parameters.categorical_hyperparameter import CategoricalHyperparameter
from qewton.optim.parameters.number_hyperparameter import DiscreteHyperparameter

from qewton.data.dataloaders.sampler.point_sampler import PointSampler


class GridSampler(PointSampler[TensorType]):
    """Samples points in a grid-like fashion from a geometry.

    Args:
        geometry (Geometry): The geometry in which points should be sampled.
        n_points (int | Hyperparameter): The number of points that should be sampled.
        compute_normals (bool, optional): Whether to compute normals to each sampled
            points. This is only possible for BoundaryGeometries. Defaults to False.
        normal_name (str | Variable): The name for the output port of the normals.
            Defaults to "normals".
        name (str, optional): The name of the node. Defaults to "PointSampler".
        state (NodeState, optional): The state of the node.
            Defaults to NodeState.FIXED.
    """

    def __init__(
        self,
        geometry: Geometry,
        n_points: int | DiscreteHyperparameter | CategoricalHyperparameter,
        compute_normals: bool = False,
        normal_name: str | Variable = "normals",
        name: str = "PointSampler",
        state: NodeState = NodeState.FIXED,
        backend: type[Backend[TensorType]] | None = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(
            geometry, n_points, compute_normals, normal_name, name, state, backend
        )
        self.is_static = True
        self.point_cache: TensorType | None = None
        self.normal_cache: TensorType | None = None

    def sample_points(self):
        """Samples points from the geometry in a grid-like fashion."""
        if self.point_cache is not None:
            return self.point_cache, self.normal_cache
        if self.is_boundary_geometry:
            self.point_cache, self.normal_cache = self.geometry.sample_grid(
                self.batch_size,
                device=self._device,
                include_normals=self.compute_normals,  # type: ignore
            )
        else:
            self.point_cache, self.normal_cache = (
                self.geometry.sample_grid(self.batch_size, device=self._device),
                None,
            )
        return self.point_cache, self.normal_cache
