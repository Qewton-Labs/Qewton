from __future__ import annotations
from abc import abstractmethod
from contextlib import contextmanager
import inspect
from typing import Callable

import numpy as np

from qewton.config.devices import Device
from qewton.data.dataloaders.base import DataNode
from qewton.geometries.base import Geometry, BoundaryGeometry
from qewton.geometries.discrete.sampled_geometry import SampledGeometry
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


@contextmanager
def discretization_mode(samplers, max_vertex_distance, device: Device | str | None = None):
    """Switch samplers to mesh mode for one run.

    `device` is only where the freshly-generated mesh points themselves are
    built (Geometry.create_mesh()'s own `device` argument defaults to cpu,
    regardless of wherever the sampler/model actually live) - it does not
    move the sampler or anything else; pass an explicit `device` to
    Graph.visualize() for that.
    """
    for s in samplers:
        s.set_mesh_mode(max_vertex_distance, device)
    try:
        yield
    finally:
        for s in samplers:
            s.unset_mesh_mode()


@contextmanager
def active_discretization_mode(samplers, geometry):
    """Evaluate `samplers` at `geometry`'s own points for one run (used by
    `graph.visualize(port, reference=...)`), instead of whatever they would
    otherwise sample or mesh. Every sampler gets the same
    geometry, mirroring discretization_mode()'s uniform mesh-mode
    injection - there is no per-sampler override for a multi-sampler graph.
    """
    for s in samplers:
        s.active_discretization = geometry
    try:
        yield
    finally:
        for s in samplers:
            s.active_discretization = None


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
        self.sampled_geometry = SampledGeometry(self.geometry, n_points)

        # clear automatically build ports:
        self._output_ports = []
        self._build_port(self.geometry.variable)
        if self.compute_normals:
            if isinstance(normal_name, str):
                self.normal_name = Variable(name=normal_name, dim=self.geometry.dim)
            else:
                self.normal_name = normal_name
            self._build_port(self.normal_name)

        self.point_cache: TensorType | None = None
        self.normal_cache: TensorType | None = None
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

        self.mesh_mode = False  # used as a mode for plotting
        self.current_mesh_max_vertex_distance = None
        self.current_mesh_device: Device | str | None = None
        self._active_discretization: Geometry | None = None

    @property
    def active_discretization(self) -> Geometry | None:
        """The geometry to evaluate this sampler's model at for one run,
        overriding both normal sampling and mesh mode - injected by
        `graph.visualize(port, reference=...)` so the model is evaluated
        exactly where a loaded reference already has data, rather than
        matching the reference onto the model's own points (which would
        leak interpolation error into the comparison).
        """
        return self._active_discretization

    @active_discretization.setter
    def active_discretization(self, geometry: Geometry | None) -> None:
        if geometry is not None:
            self._validate_active_discretization(geometry)
        self._active_discretization = geometry

    def _validate_active_discretization(self, geometry: Geometry) -> None:
        """Dimension is a hard requirement - object identity between
        `geometry.variable` and this sampler's own `self.geometry.variable`
        is preferred (reusing one Variable instance end to end avoids any
        matching problem at all), but not itself checked here. The
        bounding-box and containment checks catch a badly-matched reference
        before it ever reaches the model, rather than producing an error
        plot that looks like a bad model rather than a units/domain bug.
        """
        own_variable = self.geometry.variable
        if geometry.variable.dim != own_variable.dim:
            raise ValueError(
                f"active_discretization's variable has dim="
                f"{geometry.variable.dim}, but this sampler's domain is "
                f"{own_variable.dim}D - they must match. Reuse the same "
                "Variable instance building both geometries."
            )

        domain_box = np.asarray(self.backend.to_numpy(self.geometry.bounding_box()))
        geometry_box = np.asarray(self.backend.to_numpy(geometry.bounding_box()))
        domain_min, domain_max = domain_box[0::2], domain_box[1::2]
        geometry_min, geometry_max = geometry_box[0::2], geometry_box[1::2]
        if np.any(geometry_max < domain_min) or np.any(geometry_min > domain_max):
            raise ValueError(
                f"active_discretization's bounding box ({geometry_box}) does "
                f"not overlap this sampler's domain ({domain_box}) on at "
                "least one axis - this usually means a units/normalization "
                "mismatch (e.g. a model trained on normalized coordinates "
                "against a reference in physical units), not a genuinely "
                "out-of-domain reference."
            )

        points = geometry.discretization_points
        if points is not None:
            inside = self.backend.to_numpy(self.geometry.contains(points))
            if not np.all(inside):
                raise ValueError(
                    "active_discretization has points outside this "
                    "sampler's domain - evaluating the model there would "
                    "extrapolate, making the resulting error plot look "
                    "dramatic for the wrong reason. Restrict the reference "
                    'to the sampler\'s domain, or use on="mesh" instead.'
                )

    def set_mesh_mode(
        self,
        max_vertex_distance: float | None = None,
        device: Device | str | None = None,
    ):
        self.mesh_mode = True
        self.current_mesh_max_vertex_distance = max_vertex_distance
        # Falls back to this sampler's own current device (self._device,
        # kept up to date by DataNode.to()) rather than Geometry.create_mesh()'s
        # own cpu default, so mesh mode lands on wherever the sampler/model
        # already live unless a call site (Graph.visualize()) overrides it.
        self.current_mesh_device = device if device is not None else self._device

    def unset_mesh_mode(self):
        self.mesh_mode = False

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
            GeometryAxes(self.sampled_geometry, (AxesDim(self.batch_size),)),
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

    def forward(self) -> TensorType | tuple[TensorType, TensorType | None]:
        """Executes the data loading for one batch.

        This method handles split indexing, batch slicing, and moving data to
        the appropriate device.
        """
        if self._active_discretization is not None:
            if self.compute_normals:
                raise NotImplementedError(
                    "active_discretization currently cannot produce normals"
                )
            geometry = self._active_discretization
            points = geometry.discretization_points
            mesh = getattr(geometry, "mesh", None)

            # for plotting: store the current points and cells
            self.sampled_geometry.set_current_discretization(
                points, mesh.cells if mesh is not None else None
            )
            return points

        if self.mesh_mode:
            mesh = self.sampled_geometry.visualization_mesh(
                self.current_mesh_max_vertex_distance, self.current_mesh_device
            )
            points = mesh.vertices

            # for plotting: store the current points and cells
            self.sampled_geometry.set_current_discretization(points, mesh.cells)

            if self.compute_normals:
                raise NotImplementedError("Mesh mode currently cannot produce normals")
            return points

        # Use the cache
        if self.created_cache:
            point_slice = slice(self.cache_idx, self.cache_idx + 1)
            # Update the index for next time:
            self.cache_idx += 1
            if self.cache_idx >= len(self.point_cache):  # type: ignore
                self.cache_idx = 0
            # Take a slice and remove the first axis by taking [0]
            points = self.point_cache[point_slice][0]  # type: ignore

            # for plotting: store the current points and cells
            self.sampled_geometry.set_current_discretization(points, None)

            if self.compute_normals:
                normals = self.normal_cache[point_slice][0]  # type: ignore
                return points, normals
            return points

        # Sample points on the fly
        points, normals = self.sample_points()

        # for plotting: store the current points and cells
        self.sampled_geometry.set_current_discretization(points, None)

        if self.compute_normals:
            return points, normals
        return points

    def to(self, device: str | Device):
        super().to(device)
        # Not gated on created_cache: a subclass (e.g. GridSampler) can
        # populate point_cache/normal_cache on its own, outside the
        # cache()/created_cache opt-in mechanism - whatever's actually
        # cached needs to move with the sampler regardless of how it got
        # there, or it silently keeps serving stale-device tensors forever.
        if self.point_cache is not None:
            self.point_cache = self.backend.to(self.point_cache, self._device)
        if self.normal_cache is not None:
            self.normal_cache = self.backend.to(self.normal_cache, self._device)

    def __mul__(self, other):
        from .product_sampler import ProductSampler

        return ProductSampler(self, other)
