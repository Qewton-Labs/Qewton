"""Base class :class:`TablePlot` for plots of tables: named columns, one
row per record, e.g. the runs of a hyperparameter tuning."""

import itertools
from dataclasses import dataclass

import numpy as np

from qewton.visualization.axes import ControlAxis, Selector
from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.color_channel import candidate_scales, unique
from qewton.visualization.plots.result import Result
from qewton.visualization.references import title_of
from qewton.visualization.scales import ColorScale, Scale, log_ticks


@dataclass(frozen=True)
class TableColumn:
    """A table column as plotted: numbers, with categories coded.

    Args:
        values: One number per row; the code of its category for a
            categorical column.
        categories: The category of each code; None for a numeric column.
    """

    values: np.ndarray
    categories: list[str] | None = None


def table_columns(table) -> dict[str, TableColumn]:
    """The columns of a table, with categories coded.

    Text and object columns become categorical (codes in sorted order of
    the categories), as do boolean ones (``False`` 0, ``True`` 1), since
    renderers draw codes rather than booleans.

    Args:
        table: Anything whose ``items()`` gives (name, values) pairs, e.g.
            a dict of arrays or a pandas DataFrame.

    Returns:
        dict: ``{name: TableColumn}``.

    Raises:
        ValueError: If the columns differ in length.
    """
    columns = {}
    for name, values in table.items():
        values = np.asarray(values)
        if values.dtype.kind == "b":
            columns[name] = TableColumn(values.astype(int), ["False", "True"])
        elif values.dtype.kind in "OUS":
            categories, codes = np.unique(values.astype(str), return_inverse=True)
            columns[name] = TableColumn(codes, [str(c) for c in categories])
        else:
            columns[name] = TableColumn(values.astype(float))
    lengths = {len(c.values) for c in columns.values()}
    if len(lengths) > 1:
        raise ValueError(f"The columns of a table differ in length: {sorted(lengths)}.")
    return columns


class TablePlot(Plot):
    """Draws the rows of a table; channels are bound to column names, or
    to Selectors of column names.

    Controls reference a column: their states are its distinct values, and
    only the rows with the value of the current state are drawn (a
    FacetAxis draws one cell per value). Log columns are drawn by the
    log10 of their values, with ticks labeled by the values; categorical
    columns by their codes, with ticks labeled by the categories.

    Args:
        table: Anything whose ``items()`` gives (name, values) pairs, e.g.
            a dict of arrays or a pandas DataFrame.
        color: TableColumn name or Selector of column names colored by; None
            for a uniform color.
        color_scale: As for
            :class:`~qewton.visualization.plots.data.base.DataPlot`; a new
            scale of a log column is logarithmic.
        log_columns: Numeric columns drawn on a log scale.
        controls: Control axes, each referencing a column.
        title, label: As for :class:`~qewton.visualization.plots.base.Plot`.

    Raises:
        ValueError: If a reference names no column, a log column is not
            numeric and positive, or a control does not reference a
            column.

    Attributes:
        columns (dict): ``{name: TableColumn}``.
        n_rows (int): Number of rows.
        log_columns (set of str): The log columns.
        color_scale (ColorScale or None): Scale of the color channel in the
            current states.

    Notes:
        Subclasses implement `dim`, `positions` and :meth:`build`.
    """

    def __init__(
        self,
        table,
        *,
        color: str | Selector | None = None,
        color_scale: ColorScale | list[ColorScale] | None = None,
        log_columns: tuple[str, ...] | list[str] = (),
        controls: tuple[ControlAxis, ...] | list[ControlAxis] = (),
        title: str | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__(title=title, label=label, controls=controls)
        self.columns = table_columns(table)
        self.n_rows = len(next(iter(self.columns.values())).values) if self.columns else 0
        self.log_columns = set(log_columns)
        self._color = color
        self._check_log_columns()
        self._check_reference(color)
        self._color_scales = candidate_scales(
            color, color_scale, lambda c: ColorScale(log=c in self.log_columns)
        )
        self._resolve_controls()

    # --- references ----------------------------------------------------------

    def _check_reference(self, reference) -> None:
        if reference is None:
            return
        names = reference.candidates if isinstance(reference, Selector) else [reference]
        self._check_names(names)

    def _check_names(self, names) -> None:
        unknown = [n for n in names if n not in self.columns]
        if unknown:
            raise ValueError(f"{unknown} are no columns; the columns are {list(self.columns)}.")

    def _check_log_columns(self) -> None:
        self._check_names(sorted(self.log_columns))
        for name in self.log_columns:
            column = self.columns[name]
            if column.categories is not None or np.any(column.values <= 0):
                raise ValueError(f"Log column {name!r} must be numeric and positive.")

    def _resolve_controls(self) -> None:
        for control in self.controls:
            name = control.variable_or_axes
            if not isinstance(name, str) or name not in self.columns:
                raise ValueError(
                    f"{type(control).__name__} {control.name!r} of a table plot must "
                    f"reference a column; the columns are {list(self.columns)}."
                )
            column = self.columns[name]
            values = np.unique(column.values)
            if column.categories is not None:
                control.resolve(len(values), labels=[column.categories[int(v)] for v in values])
            else:
                control.resolve(len(values), coordinates=values)

    @staticmethod
    def current(reference) -> str | None:
        """The column a reference names in the current states.

        Args:
            reference: A column name, a Selector of them, or None.

        Returns:
            str or None: The column name.
        """
        return reference.selected if isinstance(reference, Selector) else reference

    # --- columns -------------------------------------------------------------

    def drawn_values(self, name: str, rows: np.ndarray) -> np.ndarray:
        """The values of a column as drawn on a position axis: log10 for a
        log column, codes for a categorical one.

        Args:
            name: The column.
            rows: Boolean mask of the drawn rows.

        Returns:
            numpy.ndarray: One value per drawn row.
        """
        values = self.columns[name].values[rows]
        return np.log10(values) if name in self.log_columns else values

    def column_ticks(self, name: str) -> tuple[list[float], list[str]] | None:
        """The ticks of a column on a position axis, in drawn values.

        Args:
            name: The column.

        Returns:
            tuple or None: Positions and labels: the categories, or values
                of a log column; None for automatic ticks.
        """
        column = self.columns[name]
        if column.categories is not None:
            return list(range(len(column.categories))), list(column.categories)
        if name in self.log_columns:
            log_values = np.log10(column.values)
            return log_ticks(float(log_values.min()), float(log_values.max()))
        return None

    # --- states and copies ---------------------------------------------------

    def _rows(self, overlay_states: dict) -> np.ndarray:
        # Rows with the value of each control's state.
        rows = np.ones(self.n_rows, dtype=bool)
        for control in self.controls:
            column = self.columns[control.variable_or_axes].values
            state = overlay_states.get(control, control.state)
            rows &= column == np.unique(column)[state]
        return rows

    def _copies(self):
        # {overlay control: state} per drawn copy.
        overlays = [c for c in self.controls if c.copies]
        for states in itertools.product(*(c.states for c in overlays)):
            yield dict(zip(overlays, states))

    def _trained_copies(self):
        # Every copy in every trained state; restores the states afterwards.
        stepping = [c for c in self.controls if not c.copies] + self.selectors
        originals = [c.state for c in stepping]
        try:
            for states in itertools.product(*(c.trained_states for c in stepping)):
                for control, state in zip(stepping, states):
                    control.state = state
                yield from self._copies()
        finally:
            for control, state in zip(stepping, originals):
                control.state = state

    # --- Plot interface ------------------------------------------------------

    @property
    def positions(self) -> dict:
        """``{channel: column name or Selector}`` of the position channels
        (class-specific)."""
        raise NotImplementedError

    @property
    def selectors(self) -> list[Selector]:
        references = [*self.positions.values(), self._color]
        return unique([r for r in references if isinstance(r, Selector)])

    @property
    def n_copies(self) -> int:
        return len(list(self._copies()))

    @property
    def color_scale(self) -> ColorScale | None:
        if not self._color_scales:
            return None
        index = self._color.state if isinstance(self._color, Selector) else 0
        return self._color_scales[index]

    @property
    def color_scales(self) -> list[ColorScale]:
        return unique(self._color_scales)

    @property
    def active_color_scales(self) -> list[ColorScale]:
        return [self.color_scale] if self.color_scale is not None else []

    @property
    def color_title(self) -> str | None:
        return title_of(self.current(self._color)) if self._color is not None else None

    def channels_of(self, selector: Selector) -> tuple[str, ...]:
        """See :meth:`~qewton.visualization.plots.base.Plot.channels_of`."""
        channels = [c for c, r in self.positions.items() if r is selector]
        return tuple(channels + (["color"] if self._color is selector else []))

    def axis_title(self, channel: str) -> str | None:
        """See :meth:`~qewton.visualization.plots.base.Plot.axis_title`."""
        return title_of(self.current(self.positions[channel]))

    def axis_ticks(self, channel: str) -> tuple[list[float], list[str]] | None:
        """See :meth:`~qewton.visualization.plots.base.Plot.axis_ticks`."""
        return self.column_ticks(self.current(self.positions[channel]))

    def train_scales(self, axes: dict[str, Scale]) -> None:
        """See :meth:`~qewton.visualization.plots.base.Plot.train_scales`.

        Position axes bound to a Selector are not trained, so their range
        follows the chosen column."""
        for overlay_states in self._trained_copies():
            rows = self._rows(overlay_states)
            for channel, reference in self.positions.items():
                if not isinstance(reference, Selector):
                    axes[channel].train(self.drawn_values(reference, rows))
            if self.color_scale is not None:
                self.color_scale.train(self.columns[self.current(self._color)].values[rows])

    def evaluate(self) -> list[Result]:
        """See :meth:`~qewton.visualization.plots.base.Plot.evaluate`."""
        results = []
        for index, overlay_states in enumerate(self._copies()):
            result = self.build(self._rows(overlay_states))
            parts = [self.label] + [f"{c.name} = {c.label(s)}" for c, s in overlay_states.items()]
            result.label = ", ".join(p for p in parts if p) or None
            result.color_index = self.color_index + index
            if self.color_scale is not None:
                result.color_scale = self.color_scale
                result.color_title = self.color_title
            results.append(result)
        return results

    def color_values(self, rows: np.ndarray) -> np.ndarray | None:
        """The values of the color column of the drawn rows, as given.

        Args:
            rows: Boolean mask of the drawn rows.

        Returns:
            numpy.ndarray or None: None without a color binding.
        """
        if self._color is None:
            return None
        return self.columns[self.current(self._color)].values[rows]

    def build(self, rows: np.ndarray) -> Result:
        """The Result of the drawn rows, without the common fields (label,
        colors), which :meth:`evaluate` sets.

        Args:
            rows: Boolean mask of the drawn rows.

        Returns:
            Result: The Result.
        """
        raise NotImplementedError
