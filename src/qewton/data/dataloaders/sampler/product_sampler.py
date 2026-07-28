from qewton.config.devices import Device
from qewton.graphs.nodes import OutputPort
from qewton.backends import TensorType
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import FeatureAxes, GeometryAxes
from qewton.config.variables import Variable


from qewton.data.dataloaders.sampler.point_sampler import PointSampler


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
