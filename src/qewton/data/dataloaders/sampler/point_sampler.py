"""Base class for samplers that draw points from a geometry."""

from __future__ import annotations
from abc import abstractmethod
from contextlib import contextmanager
import inspect
from typing import Callable

import numpy as np

from qewton.config.devices import Device
from qewton.data.dataloaders.base import DataNode
from qewton.geometries.base import Geometry, BoundaryGeometry, vertex_distance_for
from qewton.geometries.product import ProductGeometry
from qewton.geometries.discrete.sampled_geometry import SampledGeometry
from qewton.geometries.discrete.product_mesh_geometry import ProductMeshGeometry
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
def discretization_mode(
    samplers, max_vertex_distance, device: Device | str | None = None
):
    """Context manager that switches samplers to mesh mode.

    Inside the context, every sampler outputs the vertices of a mesh of its
    geometry, see :meth:`PointSampler.set_mesh_mode`.

    Args:
        samplers (iterable of PointSampler): The samplers to switch.
        max_vertex_distance (float or dict or None): The mesh resolution, one
            value or one per geometry variable.
        device: The device the mesh points are built on. Defaults to each
            sampler's own device. The samplers themselves are not moved.
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
    """Context manager that makes samplers output a geometry's points.

    Inside the context, every sampler outputs the discretization points of
    `geometry` instead of sampling or meshing, see
    :attr:`PointSampler.active_discretization`. All samplers get the same
    geometry.

    Args:
        samplers (iterable of PointSampler): The samplers to switch.
        geometry (Geometry): A discrete geometry with
            ``discretization_points``.
    """
    for s in samplers:
        s.active_discretization = geometry
    try:
        yield
    finally:
        for s in samplers:
            s.active_discretization = None


class PointSampler(DataNode[TensorType]):
    """Base class for data nodes that sample points from a geometry.

    The node has one output port for the points, named after the geometry
    variable, and optionally one for the normals. Each forward pass
    returns, in this order of precedence: the points of
    `active_discretization`, the vertices of a mesh in mesh mode, the next
    cached batch, or newly sampled points. The current points are also
    stored in `sampled_geometry`, e.g. for plotting.

    Args:
        geometry: The geometry to sample from. Must use the same backend as the
            node.
        n_points: The number of points sampled per batch.
        filter_fn: Restricts where points are sampled, by rejection sampling.
            Its parameters are annotated with the variables (parts of the
            geometry variable) whose coordinates they receive; it returns a
            boolean mask of accepted points.
        compute_normals: Whether to also output the normal vector at each
            point. Only possible for a
            :class:`~qewton.geometries.base.BoundaryGeometry`.
        normal_name: The variable, or the name of the variable, of the normal
            output port.
        name: The name of the node.
        state: The state of the node.
        backend: The backend used for computations.

    Raises:
        ValueError: If `compute_normals` is set for a geometry that is not a
            boundary.

    Attributes:
        geometry (Geometry): The geometry to sample from.
        sampled_geometry (SampledGeometry): Holds the points of the current
            forward pass, see
            :class:`~qewton.geometries.discrete.sampled_geometry.SampledGeometry`.
        compute_normals (bool): Whether normals are output.
        normal_name (str or Variable): The variable of the normal output port.
        filter_fn (callable or None): The rejection filter.
        point_cache (TensorType or None): Cached batches of points, see
            :meth:`cache`.
        normal_cache (TensorType or None): Cached batches of normals, see
            :meth:`cache`.
        mesh_mode (bool): Whether mesh mode is active, see
            :meth:`set_mesh_mode`.
        active_discretization (Geometry or None): If set, the sampler outputs
            the discretization points of this geometry, overriding sampling and
            mesh mode. Setting it validates the geometry and raises
            :class:`ValueError` if its dimension differs from the sampler's,
            its bounding box does not overlap the sampler's domain on some
            axis, or any of its points lies outside the domain.

    Notes:
        Subclasses must implement :meth:`sample_points`.
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
        self._mesh_cache: dict[tuple, Geometry] = {}

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
        return self._active_discretization

    @active_discretization.setter
    def active_discretization(self, geometry: Geometry | None) -> None:
        if geometry is not None:
            self._validate_active_discretization(geometry)
        self._active_discretization = geometry

    def _validate_active_discretization(self, geometry: Geometry) -> None:
        # Checks dimension, bounding-box overlap and containment. Variable
        # identity is not checked.
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
            # Checked where the domain lives, so a large domain (e.g. a mesh)
            # is not copied to the points' device.
            points = self.backend.to(
                self.backend.build_tensor(points), self.geometry.device
            ).reshape(-1, points.shape[-1])
            inside = self.backend.to_numpy(self.geometry.contains(points))
            if not np.all(inside):
                raise ValueError(
                    "active_discretization has points outside this "
                    "sampler's domain - evaluating the model there would "
                    "extrapolate, making the resulting error plot look "
                    "dramatic for the wrong reason. Restrict the reference "
                    "to the sampler's domain, and check that its coordinates "
                    f"are ordered like {own_variable.name}. For a product "
                    "domain (e.g. space x parameter), pass the reference with "
                    "one GeometryAxes per factor instead - their order is "
                    "then free."
                )

    def visualization_geometry(
        self,
        max_vertex_distance: float | dict[Variable, float] | None = None,
        device: Device | str | None = None,
    ) -> Geometry:
        """Return this sampler's geometry meshed for visualization.

        The mesh is built once per distinct ``(max_vertex_distance, device)``
        pair and cached.

        Args:
            max_vertex_distance: The maximum vertex distance of the mesh. A
                dict gives one value per geometry variable: a
                :class:`~qewton.geometries.product.ProductGeometry` receives it
                as-is to mesh each factor, any other geometry uses the entry of
                its own variable.
            device: The device of the mesh. ``None`` uses the default device of
                :meth:`~qewton.geometries.base.Geometry.create_mesh`.

        Returns:
            Geometry: The meshed geometry.
        """
        cache_key = (
            frozenset(max_vertex_distance.items())
            if isinstance(max_vertex_distance, dict)
            else max_vertex_distance,
            device,
        )
        if cache_key not in self._mesh_cache:
            if not isinstance(self.geometry, ProductGeometry):
                max_vertex_distance = vertex_distance_for(
                    max_vertex_distance, self.geometry.variable
                )
            self._mesh_cache[cache_key] = (
                self.geometry.create_mesh(max_vertex_distance, device)
                if device is not None
                else self.geometry.create_mesh(max_vertex_distance)
            )
        return self._mesh_cache[cache_key]

    def set_mesh_mode(
        self,
        max_vertex_distance: float | dict[Variable, float] | None = None,
        device: Device | str | None = None,
    ):
        """Switch the sampler to mesh mode.

        In mesh mode, the forward pass outputs the vertices of a mesh of
        the geometry (see
        :meth:`visualization_geometry`)
        and stores the cells in `sampled_geometry`.

        Args:
            max_vertex_distance: The mesh resolution, one value or one per
                geometry variable.
            device: The device the mesh is built on. Defaults to the sampler's
                current device.
        """
        self.mesh_mode = True
        self.current_mesh_max_vertex_distance = max_vertex_distance
        self.current_mesh_device = device if device is not None else self._device

    def unset_mesh_mode(self):
        """Leave mesh mode and return to normal sampling."""
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
        """Return the number of points sampled per batch."""
        return self.batch_size

    def cache(self, n_batches=1):
        """Sample a number of batches once and serve them from memory.

        Later forward passes cycle through the cached batches.

        Args:
            n_batches (int, optional): The number of batches to cache (at least
                one). Defaults to 1.
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
        """Clear the cache and return to sampling on every forward pass."""
        self.point_cache = None
        self.normal_cache = None
        self.created_cache = False

    def provides_data_in_phase(self, phase: EvaluationPhase) -> bool:
        """Return whether the sampler provides data in `phase`.

        Args:
            phase: The evaluation phase.

        Returns:
            bool: Always ``True``.
        """
        return True

    @abstractmethod
    def sample_points(self) -> tuple[TensorType, TensorType | None]:
        """Sample one batch of points.

        Returns:
            tuple of TensorType: The points and the normals, or ``None`` for
                the normals if `compute_normals` is not set.
        """
        pass

    def forward(self) -> TensorType | tuple[TensorType, TensorType | None]:
        """Return the points (and normals) for one batch.

        Returns:
            TensorType or tuple of TensorType: The points, or
                ``(points, normals)`` if `compute_normals` is set. Points of
                `active_discretization` are flattened to shape
                ``(n_points, dim)`` and moved to the sampler's device; mesh
                mode points are on the device set by `set_mesh_mode`.

        Raises:
            NotImplementedError: If normals are requested together with
                `active_discretization` or mesh mode.
        """
        if self._active_discretization is not None:
            if self.compute_normals:
                raise NotImplementedError(
                    "active_discretization currently cannot produce normals"
                )
            geometry = self._active_discretization
            # The reference geometry may live on another device than this
            # sampler and the model it feeds.
            points = self.backend.to(
                self.backend.build_tensor(geometry.discretization_points), self._device
            )
            # A structured reference (e.g. a grid, or a product of point sets)
            # is evaluated as a flat batch of points; Graph.visualize()
            # restores the structure on the result.
            points = points.reshape(-1, points.shape[-1])

            # for plotting: store the current points and cells
            self._store_discretization(geometry, points)
            return points

        if self.mesh_mode:
            geometry = self.visualization_geometry(
                self.current_mesh_max_vertex_distance, self.current_mesh_device
            )
            points = (
                geometry.discretization_points
                if isinstance(geometry, ProductMeshGeometry)
                else geometry.mesh.vertices
            )
            # An already discrete source (e.g. a loaded MeshGeometry) keeps
            # its data where it is; only the points handed on are moved.
            points = self.backend.to(points, self.current_mesh_device)

            # for plotting: store the current points and cells
            self._store_discretization(geometry, points)

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

    def _store_discretization(self, geometry: Geometry, points: TensorType):
        # A product mesh is passed on as such, so its cells are only built
        # when needed.
        if isinstance(geometry, ProductMeshGeometry):
            self.sampled_geometry.set_current_discretization(points, product=geometry)
            return
        mesh = getattr(geometry, "mesh", None)
        self.sampled_geometry.set_current_discretization(
            points, mesh.cells if mesh is not None else None
        )

    @property
    def sampled_geometries(self) -> list[SampledGeometry]:
        """Every SampledGeometry this sampler updates, e.g. to convert all of
        them for plotting."""
        return [self.sampled_geometry]

    def to(self, device: str | Device):
        """Move the node, its geometry and its cached points to `device`.

        Args:
            device: The target device.
        """
        super().to(device)
        self.geometry.to(device)
        # Not gated on created_cache: subclasses (e.g. GridSampler) may fill
        # point_cache/normal_cache without cache().
        if self.point_cache is not None:
            self.point_cache = self.backend.to(self.point_cache, self._device)
        if self.normal_cache is not None:
            self.normal_cache = self.backend.to(self.normal_cache, self._device)

    def __mul__(self, other):
        """Return the product of two samplers.

        Returns:
            ProductSampler: See
                :class:`~qewton.data.dataloaders.sampler.product_sampler.ProductSampler`.
        """
        from .product_sampler import ProductSampler

        return ProductSampler(self, other)
