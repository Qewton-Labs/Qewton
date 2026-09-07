from qewton.backends import DEFAULT_DL_BACKEND, TensorType
from qewton.backends.base import ComputingBackend
from qewton.config.axes import FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.base import DiscreteGeometry
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.graphs.nodes import InputPort, Node, OutputPort


class MeshInterpolationNode(Node[TensorType]):
    """Interpolates a per-vertex mesh field onto arbitrary target points,
    linearly (barycentric) within whichever simplex contains each point.
    Points outside the mesh come back NaN - either because
    MeshGeometry.interpolate_to() didn't find a containing cell, or because
    target_geometry carries its own point_filter (GridGeometry and
    subclasses, e.g. PlaneSliceGeometry), which is masked to NaN here too.

    Generic over the target: a plane slice (PlaneSliceGeometry), a regular 3D
    resampling grid, a scattered comparison point set, or another mesh's
    vertices for mesh-to-mesh transfer all work the same way through this one
    node - only the target geometry differs, and none of that difference is
    plane-specific. See geometries.discrete.plane_slice.PlaneSliceGeometry for
    building the plane-stack target specifically (visualization plan, section
    12); this node has no knowledge of planes at all.

    The source mesh geometry and target geometry are structural constants,
    fixed at construction time - like PointSampler.geometry, not something
    that varies per forward pass. Only the per-vertex field VALUES flow
    through the graph dynamically (e.g. a trained solution evaluated at each
    mesh vertex), so the output DataConfiguration is fully known here and
    built manually - the same pattern PointSampler/ConcatVariables use for
    ports whose config depends on constructor parameters rather than
    forward()'s type hints.

    Args:
        mesh_geometry (MeshGeometry): The source volumetric mesh.
        field_variable (Variable): The per-vertex field variable expected on
            the input port, and carried through unchanged to the output.
        target_geometry (DiscreteGeometry): Where to interpolate to - any
            geometry with discretization_points, of shape
            (*target_geometry.shape, mesh_geometry.variable.dim).
    """

    def __init__(
        self,
        mesh_geometry: MeshGeometry,
        field_variable: Variable,
        target_geometry: DiscreteGeometry,
        name: str | None = None,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        assert backend == mesh_geometry.backend, (
            f"MeshInterpolationNode and mesh_geometry should use the same backend, "
            f"found {backend} and {mesh_geometry.backend}"
        )
        self.mesh_geometry = mesh_geometry
        self.field_variable = field_variable
        self.target_geometry = target_geometry

        super().__init__(name=name or "MeshInterpolationNode", backend=backend)
        self.backend: type[ComputingBackend[TensorType]] = backend

        # Replace the ports _build_ports() inferred from forward()'s (bare,
        # untyped) signature - the real configs depend on constructor
        # parameters, not on forward()'s type hints.
        self._input_ports = [
            InputPort(
                DataConfiguration(
                    GeometryAxes(mesh_geometry), FeatureAxes(variable=field_variable)
                ),
                node=self,
                name="field",
            )
        ]
        self._output_ports = [
            OutputPort(
                DataConfiguration(
                    GeometryAxes(target_geometry), FeatureAxes(variable=field_variable)
                ),
                node=self,
                name="interpolated",
            )
        ]

    def forward(self, field):
        dim = self.mesh_geometry.variable.dim
        points = self.backend.math.reshape(
            self.target_geometry.discretization_points, (-1, dim)
        )
        interpolated = self.mesh_geometry.interpolate_to(points, field)
        feature_shape = tuple(self.backend.math.shape(interpolated))[1:]
        interpolated = self.backend.math.reshape(
            interpolated, (*self.target_geometry.shape, *feature_shape)
        )

        point_filter = getattr(self.target_geometry, "point_filter", None)
        if point_filter is not None:
            # point_filter's shape is (*target_geometry.shape, 1) - reshape
            # its trailing axis to one singleton per feature dim instead, so
            # it broadcasts against `interpolated` whether the field is
            # scalar (no trailing axis at all) or multi-component.
            mask = self.backend.math.reshape(
                point_filter, self.target_geometry.shape + (1,) * len(feature_shape)
            )
            nan = self.backend.build_tensor(float("nan"))
            interpolated = self.backend.math.where(
                self.backend.math.logical_not(mask), nan, interpolated
            )

        return interpolated
