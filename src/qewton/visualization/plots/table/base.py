import numpy as np

from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.result import Column
from qewton.visualization.plots.spec import ControlSpec


class TablePlot(Plot):
    """A Plot whose values come from named columns rather than `data` +
    `DataConfiguration` - the input family for tuner/log analysis (see the
    implementation plan, section 1, "The input-family boundary").

    This data was never graph-tensor data: it comes from run logs, is a flat
    table of named columns, some numeric and some categorical, with no
    geometry and no composed feature vectors. Forcing it through
    `DataConfiguration` was tried and discarded (plan, section 11) - it loses
    categorical labels before the plot ever sees them and describes log data
    as if it had tensor axis semantics it doesn't have.
    """

    def __init__(
        self,
        columns,
        title=None,
        theme=None,
        controls: list[ControlSpec] | None = None,
    ) -> None:
        super().__init__(title=title, theme=theme, controls=controls)
        self.columns = self._normalize(columns)
        self.n_rows = len(next(iter(self.columns.values()))) if self.columns else 0

        for spec in self.controls:
            values = sorted(set(self.columns[spec.variable_or_axes].values.tolist()))
            spec.resolve(values)

    @staticmethod
    def _normalize(columns) -> dict[str, Column]:
        """Accepts a mapping of arrays, a pandas DataFrame, or the tuner's own
        result object - anything whose `.items()` yields (name, array-like)
        pairs, which a dict and a DataFrame both do. Normalizes to
        dict[str, Column]. Categorical columns (non-numeric dtype) are coded
        to integers exactly once, here, so labels survive to the artist
        instead of being flattened by the caller."""
        normalized = {}
        for name, col in columns.items():
            if isinstance(col, Column):
                normalized[name] = col
                continue
            values = np.asarray(col)
            if values.dtype.kind in "OU":  # object/unicode -> categorical
                labels, codes = np.unique(values, return_inverse=True)
                normalized[name] = Column(values=codes, labels=list(labels))
            else:
                normalized[name] = Column(values=values)
        return normalized

    def apply_controls(self) -> dict[str, Column]:
        """Controls select rows here, rather than indexing a tensor
        dimension - same declaration (ControlSpec/spec.state) as DataPlot,
        different reduction. Row-filtering controls change the *number* of
        rows between states, so Artist.update() replaces whole arrays rather
        than swapping one value."""
        mask = np.ones(self.n_rows, dtype=bool)
        for spec in self.controls:
            mask &= self.columns[spec.variable_or_axes].values == spec.state
        return {name: col[mask] for name, col in self.columns.items()}
