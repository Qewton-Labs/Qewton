from __future__ import annotations
from abc import abstractmethod
import inspect
from typing import Callable

from qewton.config.devices import Device
from qewton.data.dataloaders.base import DataNode
from qewton.geometries.base import Geometry, BoundaryGeometry
from qewton.graphs.nodes import NodeState, OutputPort
from qewton.backends import DEFAULT_DL_BACKEND, TensorType, ComputingBackend
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import AxesDim, FeatureAxes, GeometryAxes  # BatchAxes
from qewton.config.variables import Variable
from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
)
from qewton.optim.parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
)


class PointSampler(DataNode[TensorType]):
    """Parent class for sampling individual points from a geometry.

    Args:
        geometry (Geometry): The geometry in which points should be sampled.
        n_points (int | Hyperparameter): The number of points that should be sampled.
        filter_fn (callable, optional): An additional filter that specifies at which
            location points should be sampled. Internally a rejection sampling
            strategy is used. Default is None.
        compute_normals (bool, optional): Whether to compute normals to each sampled
            points. This is only possible for BoundaryGeometries. Defaults to False.
        normal_name (str | Variable): The name for the output port of the normals.
            Defaults to "normals".
        name (str, optional): The name of the node. Defaults to "PointSampler".
        state (NodeState, optional): The state of the node.
            Defaults to NodeState.FIXED.
        backend (type[ComputingBackend[TensorType]], optional): What backend the node
            should use for computations, etc. Defaults to the deep learning
            backend (DEFAULT_DL_BACKEND).
    """

    def __init__(
        self,
        geometry: Geometry,
        n_points: int | DiscreteHyperparameter | CategoricalHyperparameter,
        filter_fn: Callable | None = None,
        compute_normals: bool = False,
        normal_name: str | Variable = "normals",
        name: str = "PointSampler",
        state: NodeState = NodeState.FIXED,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        assert backend == geometry.backend, (
            f"Sampler and geometry should use the same backend, found {backend} "
            + f"and {geometry.backend}"
        )
        self.geometry = geometry
        self.compute_normals = compute_normals
        self.normal_name = normal_name
        self.has_boundary_geometry = isinstance(geometry, BoundaryGeometry)
        self.is_static = False
        if compute_normals:
            self._check_normal_sampling_possible()

        super().__init__(batch_size=n_points, name=name, state=state, backend=backend)
        self.backend: type[ComputingBackend[TensorType]] = backend

        # clear automatically build ports:
        self._output_ports = []
        self._build_port(self.geometry.variable)
        if self.compute_normals:
            if isinstance(normal_name, str):
                self.normal_name = Variable(name=normal_name, dim=self.geometry.dim)
            else:
                self.normal_name = normal_name
            self._build_port(self.normal_name)

        self.point_cache: list[TensorType] = []
        self.normal_cache: list[TensorType | None] = []
        self.created_cache: bool = False
        self.cache_idx: int = 0

        self.filter_fn = filter_fn
        self.filter_indices: list[tuple[slice, ...]] | list[list[int]] = []
        # build the indices
        if filter_fn is not None:
            sig = inspect.signature(filter_fn).parameters.values()
            for var in sig:
                if isinstance(var.annotation, Variable):
                    self.filter_indices.append(
                        self.geometry.variable.get_slice(var.annotation)  # type: ignore
                    )

    def _check_normal_sampling_possible(self):
        if not self.has_boundary_geometry:
            raise ValueError(
                f"{self.geometry} is not a boundary geometry, can not compute normals."
            )

    def _evaluate_filter(self, points):
        filter_output = self.filter_fn(  # type: ignore
            *(points[..., indices] for indices in self.filter_indices)
        )
        return self.backend.math.where(filter_output)[0]

    def _build_port(self, variable: Variable):
        axes = [
            # BatchAxes(AxesDim(self.batch_size)),
            GeometryAxes(self.geometry, (AxesDim(self.batch_size),)),
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

    def cache(self, n_batches=1):
        """Caches a number of sampling points, such that they are only computed
        once at the beginning and afterwards read from memory. Can be useful
        when data computation is expensive or always the same points should be
        used.

        Args:
            n_batches (int, optional): How many points should be cached.
                The number n_batches is in respect to the batch_size. Defaults to 1,
                hence one batch of points is sampled.
        """
        run_sampling = max(n_batches, 1)
        self.point_cache, self.normal_cache = [], []
        for _ in range(run_sampling):
            points, normals = self.sample_points()
            self.point_cache.append(points)
            if self.compute_normals:
                self.normal_cache.append(normals)
        self.created_cache = True
        self.cache_idx = 0

    def clear_cache(self):
        """Clears the cache and goes back to *online* sampling."""
        self.point_cache = []
        self.normal_cache = []
        self.created_cache = False

    def provides_data_in_phase(self, phase: EvaluationPhase) -> bool:
        return True

    @abstractmethod
    def sample_points(self) -> tuple[TensorType, TensorType | None]:
        pass

    def forward(self) -> TensorType | tuple[TensorType, TensorType | None]:
        """Executes the data loading for one batch.

        This method handles split indexing, batch slicing, and moving data to
        the appropriate device.
        """
        # Use the cache
        if self.created_cache:
            # point_slice = slice(self.cache_idx, self.cache_idx + 1)
            # Take a slice and remove the first axis by taking [0]
            points = self.point_cache[self.cache_idx]
            normals = None
            if self.compute_normals:
                normals = self.normal_cache[self.cache_idx]

            # Update the index for next time:
            self.cache_idx += 1
            if self.cache_idx >= len(self.point_cache):
                self.cache_idx = 0

            if self.compute_normals:
                return points, normals
            return points

        # Sample points on the fly
        points, normals = self.sample_points()
        if self.compute_normals:
            return points, normals
        return points

    def to(self, device: str | Device):
        super().to(device)
        if self.created_cache:
            self.point_cache = [
                self.backend.to(p, self._device) for p in self.point_cache
            ]
            if self.compute_normals:
                self.normal_cache = [
                    self.backend.to(n, self._device) for n in self.normal_cache
                ]

    def __mul__(self, other):
        from .product_sampler import ProductSampler

        return ProductSampler(self, other)
