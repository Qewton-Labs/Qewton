import numpy as np

from qewton.config.axes import Axes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.result import CurveResult, PathResult
from qewton.visualization.plots.spec import AxisSpec, ControlSpec, VectorSpec


class LinePlot(Plot):
    """A single curve: one value plotted over one structural domain axis.

    Three fields over a 1D geometry are three separate LinePlots, not one
    plot with three lines - each evaluate() stays unambiguous about what it
    returns, same reasoning as "no universal evaluate()" for the rest of the
    hierarchy. Multiple LinePlots in one Figure share axis ranges for free
    (see the "Axis ranges" note in the Facets section of the plan) - no
    special handling needed here.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,
        y: AxisSpec | Variable,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, controls=controls)
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

        # x falls back to plain indices until "Tick coordinates" lands.
        x_values = np.arange(y_values.shape[0])
        return CurveResult(x=x_values, y=y_values)

    def create_artist(self, backend_figure, renderer):
        return renderer.LineArtist.create(backend_figure, self)


class PathPlot(Plot):
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

    def create_artist(self, backend_figure, renderer):
        return renderer.PathArtist.create(backend_figure, self)
