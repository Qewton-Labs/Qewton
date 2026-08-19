import numpy as np

from qewton.visualization.plots.result import TableResult
from qewton.visualization.plots.spec import ColorSpec, ControlSpec
from qewton.visualization.plots.table.base import TablePlot


class ParallelCoordinatesPlot(TablePlot):
    """One vertical axis per named column, one polyline per row - the
    standard tool for inspecting hyperparameter tuning runs (which parameter
    settings produced which metric values). Replaces
    `TuningAnalyzer.plot_parallel_coordinates` (plots/tuning/analyzer.py,
    currently disabled).

    `axes` names which columns to draw, in order - typically the tunable
    parameters followed by the metrics. `color` names one more column
    (usually a metric) to color lines by; like every other family, it trains
    a shared Scale if one is attached.

    Deliberately not on this plot: `top_k` selection and minimize/maximize
    "objective" semantics from the old analyzer. Selecting a subset of rows
    produces a different dataset, so per the plan's node-layer criterion
    (section 1) it belongs upstream of construction - the caller pre-filters
    `columns` - which keeps this plot as reusable as ScatterPlot.
    """

    def __init__(
        self,
        columns,
        axes: list[str],
        color: ColorSpec | str | None = None,
        labels: dict[str, str] | None = None,
        controls: list[ControlSpec] | None = None,
        **kwargs,
    ):
        super().__init__(columns, controls=controls, **kwargs)
        self.axes = list(axes)
        missing = [key for key in self.axes if key not in self.columns]
        if missing:
            raise ValueError(
                f"{missing} not found in columns (have {list(self.columns)})."
            )
        self.labels = labels or {}

        self.color = (
            (color if isinstance(color, ColorSpec) else ColorSpec(color))
            if color is not None
            else None
        )
        if self.color is not None and self.color.variable_or_axes not in self.columns:
            raise ValueError(
                f"color column {self.color.variable_or_axes!r} not found in "
                f"columns (have {list(self.columns)})."
            )

    @property
    def embedding_dim(self) -> int | None:
        return None  # go.Parcoords lays out its own axes - no cartesian x/y

    def evaluate(self):
        rows = self.apply_controls()
        color = (
            np.asarray(rows[self.color.variable_or_axes].values)
            if self.color is not None
            else None
        )
        return TableResult(
            columns={key: rows[key] for key in self.axes}, color=color
        )

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.ParallelCoordinatesArtist.create(
            backend_figure, self, row=row, col=col
        )
