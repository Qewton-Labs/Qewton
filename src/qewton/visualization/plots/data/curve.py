import numpy as np

from qewton.config.axes import Axes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.result import CurveResult, PathResult
from qewton.visualization.plots.spec import AxisSpec, ControlSpec, VectorSpec


class LinePlot(DataPlot):
    """A single curve: one value plotted over one structural domain axis.

    To plot several fields at once, use one LinePlot per field rather than
    a single multi-line plot - each keeps its own `evaluate()` unambiguous
    about what it returns. Multiple LinePlots placed in the same Figure
    automatically share axis ranges.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,
        y: AxisSpec | Variable,
        controls: list[ControlSpec] | None = None,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)
        self.x = x if isinstance(x, AxisSpec) else AxisSpec(x)
        self.y = y if isinstance(y, AxisSpec) else AxisSpec(y)

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()

        x_idx = self._resolve_structural_dim(self.x)
        x_dim = index_map(x_idx)

        slc = self.data_config.get_variable_slice(self.y.variable_or_axes)
        y_values = np.moveaxis(np.asarray(data[slice_map(slc)]), x_dim, 0)
        if y_values.size != y_values.shape[0]:
            raise ValueError(
                f"{self.y.name} yields {y_values.size} values across "
                f"{y_values.shape[0]} steps along x ({self.x.variable_or_axes}) - "
                "remaining dimensions are unresolved. Add a SliderSpec or "
                "FixedSpec for them."
            )
        y_values = y_values.reshape(-1)

        x_values = self._geometry_x_values(y_values.shape[0])
        if x_values is None:
            # No 1D geometry behind x - the plain sample index.
            x_values = np.arange(y_values.shape[0])
        return CurveResult(x=x_values, y=y_values)

    def _geometry_x_values(self, n: int):
        """Real coordinate values for the x axis, from a 1D geometry's own
        `discretization_points` - when `x` names a GeometryAxes directly, or
        is the sole structural leaf of a 1D GridGeometry. None if there's no
        such geometry, or its points don't match `n` 1D values exactly - the
        plain sample index is always a safe fallback.
        """
        axes = self.x.variable_or_axes
        geometry = None
        if isinstance(axes, GeometryAxes):
            geometry = axes.geometry
        elif isinstance(axes, Variable):
            for candidate in self.data_config.axes:
                if isinstance(candidate, GeometryAxes) and isinstance(
                    candidate.geometry, GridGeometry
                ):
                    if axes in candidate.geometry.variable.leaves:
                        geometry = candidate.geometry
                        break
        if geometry is None:
            return None
        # A bare MeshGeometry has no discretization_points - its vertices
        # serve the same purpose.
        mesh = getattr(geometry, "mesh", None)
        raw_points = geometry.discretization_points
        if raw_points is None and mesh is not None:
            raw_points = mesh.vertices
        if raw_points is None:
            return None
        # Points may still be a backend tensor (needs backend.to_numpy()) or
        # already plain numpy (which backend.to_numpy() rejects).
        points = raw_points if isinstance(raw_points, np.ndarray) else np.asarray(
            geometry.backend.to_numpy(raw_points)
        )
        if points.ndim != 2 or points.shape[-1] != 1 or points.shape[0] != n:
            return None
        return points[:, 0]

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.LineArtist.create(backend_figure, self, row=row, col=col)


class PathPlot(DataPlot):
    """An ordered sequence of positions in space - a trajectory or streamline.

    Unlike LinePlot's three-separate-curves rule, a path's components form
    ONE position and must stay together, so PathPlot takes a single
    multi-component VectorSpec (2D or 3D) instead of separate value roles -
    same distinction MeshVectorPlot draws between a scalar field and a vector.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        position: VectorSpec | Variable,
        controls: list[ControlSpec] | None = None,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)
        self.position = (
            position if isinstance(position, VectorSpec) else VectorSpec(position)
        )

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()
        slc = self.data_config.get_variable_slice(self.position.variable_or_axes)
        positions = np.asarray(data[slice_map(slc)]).reshape(
            -1, self.position.n_dimensions
        )
        positions = self.coord_transform.apply(positions)
        return PathResult(positions=positions)

    @property
    def embedding_dim(self) -> int:
        return self.position.n_dimensions

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.PathArtist.create(backend_figure, self, row=row, col=col)
