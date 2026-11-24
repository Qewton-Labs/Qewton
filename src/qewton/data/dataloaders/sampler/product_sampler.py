"""Sampler for the Cartesian product of two samplers."""

from qewton.config.devices import Device
from qewton.graphs.nodes import OutputPort
from qewton.backends import TensorType
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import FeatureAxes, GeometryAxes
from qewton.config.variables import Variable


from qewton.data.dataloaders.sampler.point_sampler import PointSampler
from qewton.geometries.discrete.sampled_geometry import SampledGeometry


class ProductSampler(PointSampler[TensorType]):
    """Sampler that combines every point of one sampler with every point of
    another.

    Usually created with ``sampler_a * sampler_b``. The geometry is the
    product of both geometries. The point output has the geometry axes of
    `sampler_a` followed by those of `sampler_b`, and one feature axis
    holding the concatenated coordinates.

    Args:
        sampler_a: The first factor sampler.
        sampler_b: The second factor sampler. Must use the same backend and a
            different variable than `sampler_a`.
        name: The name of the node.

    Attributes:
        sampler_a (PointSampler): The first factor sampler.
        sampler_b (PointSampler): The second factor sampler.

    Notes:
        Normals are output if one of the factor samplers computes normals; that
        sampler's normals are broadcast over the other factor.
    """

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
            geometry=sampler_a.geometry * sampler_b.geometry,
            n_points=sampler_a.batch_size * sampler_b.batch_size,
            name=name,
            compute_normals=compute_normals,
            normal_name=normal_name,
            backend=sampler_a.backend,
        )

    def _check_normal_sampling_possible(self):
        if (
            not self.sampler_a.has_boundary_geometry
            and not self.sampler_b.has_boundary_geometry
        ):
            raise ValueError(
                f"{self.geometry} is not a boundary geometry, can not compute normals."
            )

    def _build_port(self, variable: Variable):
        a_config = self.sampler_a.output_ports[0].data_configuration
        b_config = self.sampler_b.output_ports[0].data_configuration
        assert (
            a_config.variables != b_config.variables
        ), f"ProductSampler can only work on samplers of different variables. \
              Found {a_config.variables} and {b_config.variables}"
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

    def set_mesh_mode(
        self,
        max_vertex_distance: float | dict[Variable, float] | None = None,
        device: Device | str | None = None,
    ):
        """Switch this sampler and both factor samplers to mesh mode.

        Args:
            max_vertex_distance: The mesh resolution, one value or one per
                geometry variable.
            device: The device the meshes are built on.
        """
        super().set_mesh_mode(max_vertex_distance, device)
        self.sampler_a.set_mesh_mode(max_vertex_distance, device)
        self.sampler_b.set_mesh_mode(max_vertex_distance, device)

    def unset_mesh_mode(self):
        """Leave mesh mode in this sampler and both factor samplers."""
        super().unset_mesh_mode()
        self.sampler_a.unset_mesh_mode()
        self.sampler_b.unset_mesh_mode()

    def forward(self):
        """Return the product points (and normals) for one batch.

        In mesh mode, the vertices of both factor meshes are combined.

        Returns:
            TensorType or tuple of TensorType: The points, or
                ``(points, normals)`` if normals are computed.
        """
        # In mesh mode both samplers mesh their own geometry, so each
        # GeometryAxes of the output refers to a meshed SampledGeometry.
        if self.mesh_mode and self._active_discretization is None:
            points, _ = self._combine(
                self.sampler_a.forward(), None, self.sampler_b.forward(), None
            )
            return points
        return super().forward()

    def sample_points(self) -> tuple[TensorType, TensorType | None]:
        """Sample both factors and combine every pair of points.

        Returns:
            tuple of TensorType: The points of shape
                ``(*geometry_shape_a, *geometry_shape_b, dim_a + dim_b)`` and
                the broadcast normals, or ``None``.
        """
        points_a, normals_a = self.sampler_a.sample_points()
        points_b, normals_b = self.sampler_b.sample_points()
        # The output's axes are those of the factors, so their geometries
        # hold the current points too, e.g. for plotting.
        self.sampler_a.sampled_geometry.set_current_discretization(points_a, None)
        self.sampler_b.sampled_geometry.set_current_discretization(points_b, None)
        return self._combine(points_a, normals_a, points_b, normals_b)

    def _combine(self, points_a, normals_a, points_b, normals_b):
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

    @property
    def sampled_geometries(self) -> list[SampledGeometry]:
        """This sampler's SampledGeometry and those of both factor samplers."""
        return [
            self.sampled_geometry,
            *self.sampler_a.sampled_geometries,
            *self.sampler_b.sampled_geometries,
        ]

    def to(self, device: str | Device):
        """Move this sampler and both factor samplers to `device`.

        Args:
            device: The target device.
        """
        self.sampler_a.to(device=device)
        self.sampler_b.to(device=device)
        super().to(device)
