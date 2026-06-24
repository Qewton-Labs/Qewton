from abc import abstractmethod

from qewton.data.dataloaders.base import DataNode
from qewton.geometries.base import Geometry, BoundaryGeometry
from qewton.graphs.nodes import NodeState, OutputPort
from qewton.backends import Backend, DEFAULT_DL_BACKEND, TensorType
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import BatchAxes, AxesDim, FeatureAxes
from qewton.config.variables import Variable
from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
)
from qewton.optim.parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
)


# TODO: Add static sampling and caching
class PointSampler(DataNode[TensorType]):
    """Parent class for sampling individual points from a geometry.

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
        assert backend == geometry.backend, (
            f"Sampler and geometry should use the same backend, found {backend} "
            + f"and {geometry.backend}"
        )
        self.geometry = geometry
        self.compute_normals = compute_normals
        self.is_boundary_geometry = isinstance(geometry, BoundaryGeometry)
        self.is_static = False
        if compute_normals and not self.is_boundary_geometry:
            raise ValueError(
                f"{geometry} is not a boundary geometry, can not compute normals."
            )
        super().__init__(batch_size=n_points, name=name, state=state, backend=backend)

        self._build_port(self.geometry.variable)
        if self.compute_normals:
            if isinstance(normal_name, str):
                normal_name = Variable(name=normal_name)
            self._build_port(normal_name)

        self.point_cache: TensorType | None = None
        self.normal_cache: TensorType | None = None

    def _build_port(self, variable: Variable):
        axes = [
            BatchAxes(AxesDim(self.batch_size)),
            # GeometryAxes(self.geometry), # TODO: How add this here???
            FeatureAxes(variable=variable),
        ]
        self._output_ports.append(
            OutputPort(
                DataConfiguration(
                    *axes,
                    dtype=self.backend.default_dtype if self.backend else None,
                ),
                node=self,
                name=variable.name,
            )
        )

    def __len__(self):
        return self.batch_size

    def cache(self, n_batches=-1):
        # TODO
        pass

    def provides_data_in_phase(self, phase: EvaluationPhase) -> bool:
        return True

    @abstractmethod
    def sample_points(self) -> tuple[TensorType, TensorType | None]:
        pass

    def forward(self):
        """Executes the data loading for one batch.

        This method handles split indexing, batch slicing, and moving data to
        the appropriate device.
        """
        points, normals = self.sample_points()
        if self.compute_normals:
            return points, normals
        return points
