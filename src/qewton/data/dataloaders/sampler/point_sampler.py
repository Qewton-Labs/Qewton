from __future__ import annotations
from abc import abstractmethod

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
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        assert backend == geometry.backend, (
            f"Sampler and geometry should use the same backend, found {backend} "
            + f"and {geometry.backend}"
        )
        self.geometry = geometry
        self.compute_normals = compute_normals
        self.normal_name = normal_name
        self.is_boundary_geometry = isinstance(geometry, BoundaryGeometry)
        self.is_static = False
        if compute_normals and not self.is_boundary_geometry:
            raise ValueError(
                f"{geometry} is not a boundary geometry, can not compute normals."
            )
        super().__init__(batch_size=n_points, name=name, state=state, backend=backend)
        self.backend: type[ComputingBackend[TensorType]] = backend

        self._build_port(self.geometry.variable)
        if self.compute_normals:
            if isinstance(normal_name, str):
                self.normal_name = Variable(name=normal_name)
            else:
                self.normal_name = normal_name
            self._build_port(self.normal_name)

        self.point_cache: TensorType | None = None
        self.normal_cache: TensorType | None = None
        self.created_cache: bool = False
        self.cache_idx: int = 0

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
        point_list, normal_list = [], []
        for _ in range(run_sampling):
            points, normals = self.sample_points()
            point_list.append(self.backend.math.unsqueeze(points, axis=0))
            if self.compute_normals:
                normal_list.append(self.backend.math.unsqueeze(normals, axis=0))
        self.point_cache = self.backend.math.concatenate(point_list, axis=0)
        if self.compute_normals:
            self.normal_cache = self.backend.math.concatenate(normal_list, axis=0)
        self.created_cache = True
        self.cache_idx = 0

    def clear_cache(self):
        """Clears the cache and goes back to *online* sampling."""
        self.point_cache = None
        self.normal_cache = None
        self.created_cache = False

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
        # Use the cache
        if self.created_cache:
            point_slice = slice(self.cache_idx, self.cache_idx + 1)
            # Update the index for next time:
            self.cache_idx += 1
            if self.cache_idx >= len(self.point_cache):  # type: ignore
                self.cache_idx = 0
            # Take a slice and remove the first axis by taking [0]
            points = self.point_cache[point_slice][0]  # type: ignore
            if self.compute_normals:
                normals = self.normal_cache[point_slice][0]  # type: ignore
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
            self.point_cache = self.backend.to(self.point_cache, self._device)
            if self.compute_normals:
                self.normal_cache = self.backend.to(self.normal_cache, self._device)

    def __mul__(self, other):
        return ProductSampler(self, other)


class ProductSampler(PointSampler[TensorType]):
    def __init__(
        self,
        sampler_a: PointSampler,
        sampler_b: PointSampler,
        name: str = "ProductSampler",
    ) -> None:
        assert sampler_a.backend == sampler_b.backend, "Backends do not fit together!"
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b

        compute_normals = False
        normal_name = "normals"
        if sampler_a.compute_normals:
            compute_normals = True
            normal_name = sampler_a.normal_name
        elif sampler_b.compute_normals:
            compute_normals = True
            normal_name = sampler_b.normal_name

        super().__init__(
            geometry=sampler_a.geometry,
            n_points=sampler_a.batch_size * sampler_b.batch_size,
            name=name,
            compute_normals=compute_normals,
            normal_name=normal_name,
            backend=sampler_a.backend,
        )

    def _build_port(self, variable: Variable):
        a_config = self.sampler_a.output_ports[0].data_configuration
        b_config = self.sampler_b.output_ports[0].data_configuration
        assert (
            a_config.variables != b_config.variables
        ), "ProductSampler can only work on samplers of different variables."
        combined_variable = a_config.variables * b_config.variables  # type: ignore
        axes = []
        for axis in a_config.axes + b_config.axes:
            if isinstance(axis, GeometryAxes):
                axes.append(axis)
        axes.append(FeatureAxes(variable=combined_variable))
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

    def sample_points(self) -> tuple[TensorType, TensorType | None]:
        points_a, normals_a = self.sampler_a.sample_points()
        points_b, normals_b = self.sampler_b.sample_points()
        # Sampler is assumed to always return points in the shape of
        # (GeometryAxes1, ..., FeatureAxes)
        a_shape = self.backend.math.shape(points_a)
        b_shape = self.backend.math.shape(points_b)
        # Now extend the sampled points such that we at the end can
        # build a tensor of the shape:
        # (GeometryAxes_a_1, ..., GeometryAxes_b_1, ..., Features_a + Features_b)
        points_a = self._add_dims(points_a, len(b_shape) - 1, -2)
        points_b = self._add_dims(points_b, len(a_shape) - 1, 0)
        new_shape = a_shape[:-1] + b_shape[:-1]
        points = self.backend.math.concatenate(
            [
                self.backend.math.broadcast_to(points_a, new_shape + (a_shape[-1],)),
                self.backend.math.broadcast_to(points_b, new_shape + (b_shape[-1],)),
            ],
            axis=-1,
        )
        # Also expand the normals
        normals = None
        if normals_a is not None:
            normals = self._expand_normals(normals_a, len(b_shape) - 1, -2, new_shape)
        elif normals_b is not None:
            normals = self._expand_normals(normals_b, len(a_shape) - 1, 0, new_shape)

        return points, normals

    def _add_dims(self, data: TensorType, times: int, idx: int):
        for _ in range(times):
            data = self.backend.math.unsqueeze(data, idx)
        return data

    def _expand_normals(
        self, normals: TensorType, times: int, idx: int, new_shape: tuple[int, ...]
    ):
        n_dim = self.backend.math.shape(normals)[-1]
        normals = self._add_dims(normals, times, idx)
        return self.backend.math.broadcast_to(normals, new_shape + (n_dim,))

    def to(self, device: str | Device):
        self.sampler_a.to(device=device)
        self.sampler_b.to(device=device)
        super().to(device)
