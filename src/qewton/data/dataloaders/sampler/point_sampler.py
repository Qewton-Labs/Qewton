from abc import abstractmethod
import numpy as np

from ..base import DataNode
from ....geometries.base import Geometry, BoundaryGeometry
from ....graphs.nodes import NodeState, OutputPort
from ....config.backend import Backend, DEFAULT_DL_BACKEND
from ....config.data_configurations import DataConfiguration
from ....config.axes import BatchAxes, AxesDim, FeatureAxes, GeometryAxes
from ....config.variables import Variable
from ....optim.base import EvaluationPhase
from ....optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
)
from ....optim.parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
)


# TODO: Add static sampling and caching
class PointSampler(DataNode):
    """Parent class for sampling individual points from a geometry."""

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
        """
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

        self.point_cache: np.ndarray | None = None
        self.normal_cache: np.ndarray | None = None

    def _build_port(self, variable: Variable):
        axes = [
            BatchAxes(AxesDim(self.batch_size)),
            GeometryAxes(self.geometry),
            FeatureAxes(variable=variable),
        ]
        self._output_ports.append(
            OutputPort(
                DataConfiguration(
                    *axes,
                    dtype=self.backend.standard_datatype() if self.backend else None,
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
    def sample_points(self) -> tuple[np.ndarray, np.ndarray | None]:
        pass

    def forward(self):
        """Executes the data loading for one batch.

        This method handles split indexing, batch slicing, and moving data to
        the appropriate device.
        """
        points, normals = self.sample_points()
        # Transform data to backend format
        points = self.backend.from_numpy(points)
        # Move batch to device if backend is specified
        if self._device is not None and self.backend is not None:
            points = self.backend.to(points, self._device)

        if self.compute_normals:
            normals = self.backend.from_numpy(normals)
            if self._device is not None and self.backend is not None:
                normals = self.backend.to(normals, self._device)
            return points, normals
        return points
