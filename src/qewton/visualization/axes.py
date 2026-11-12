"""Axes of a drawing and controls.

Position axes (x, y, z) of an Overlay and control axes together span the
dimensions of a drawing: a plot draws the dimensions of its data that its
positions cover and reduces every other one with a control axis. A
:class:`Selector` lets the viewer choose which reference fills a channel.

"Axis" here is a dimension of the drawing; the tensor axes of a
DataConfiguration are :class:`~qewton.config.axes.Axes`.
"""

import numpy as np

from qewton.config.axes import Axes
from qewton.config.variables import Variable
from qewton.visualization.references import title_of
from qewton.visualization.scales import Scale

#: The position channels, in the order a position with several components
#: fills them.
POSITION_CHANNELS = ("x", "y", "z")


class PositionAxis:
    """A coordinate axis of an Overlay: x, y or z, by its keyword there.

    Args:
        scale: Scale of the axis; a linear one if None. Overlays sharing a
            Scale instance get one range.
        ticks: Tick positions, or None for automatic ones.
        title: Title; None derives it from the references bound to the
            axis (one shared reference: its name, different ones: none).

    Attributes:
        scale (Scale): Scale of the axis.
        ticks (list of float or None): Tick positions.
        title (str or None): The explicit title.
    """

    def __init__(
        self,
        scale: Scale | None = None,
        ticks: list[float] | None = None,
        title: str | None = None,
    ) -> None:
        self.scale = scale if scale is not None else Scale()
        self.ticks = ticks
        self.title = title


class Control:
    """Something with a state the viewer sets, or that is set for them.

    Plots receiving the same instance share its state and its widget.

    Attributes:
        state (int): The current state, an index into `states`.
        states (list of int): All states.
        trained_states (list of int): The states scales are trained over.
        name (str): Display name, e.g. for a widget label.

    Notes:
        Subclasses implement `states`, `name` and :meth:`label`.
    """

    def __init__(self, init_state: int = 0) -> None:
        self._state = init_state

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, value: int) -> None:
        if value not in self.states:
            raise ValueError(
                f"{value} is not a state of {self.name}; states: {self.states}."
            )
        self._state = value

    @property
    def states(self) -> list[int]:
        raise NotImplementedError

    @property
    def trained_states(self) -> list[int]:
        return self.states

    @property
    def name(self) -> str:
        raise NotImplementedError

    def label(self, state: int) -> str:
        """Display text of `state`, e.g. for a slider step."""
        raise NotImplementedError


class ControlAxis(Control):
    """Reduces one dimension of the data to its states.

    Every plot it is passed to resolves it against its data
    (:meth:`resolve`); plots sharing an instance must agree on the size.

    Args:
        variable_or_axes: The dimension: an Axes of one tensor axis, the
            Variable of a one-dimensional geometry (e.g. time), or the name
            of a table column (one state per value).
        init_state: Initial state.
        title: Title, e.g. of the widget; derived from `variable_or_axes`
            if None.

    Attributes:
        variable_or_axes (Variable or Axes or str): The dimension.
        coordinates (numpy.ndarray or None): Coordinate of each state, e.g.
            the time of each time step; None without coordinates.
        labels (list of str or None): Display text of each state, e.g. the
            categories of a table column; None to show the coordinates.
        copies (bool): Whether the plot draws one copy per state instead of
            one state at a time (class attribute).
    """

    copies = False

    def __init__(
        self,
        variable_or_axes: Variable | Axes | str,
        init_state: int = 0,
        title: str | None = None,
    ) -> None:
        super().__init__(init_state)
        self.variable_or_axes = variable_or_axes
        self._title = title
        self._size: int | None = None
        self.coordinates: np.ndarray | None = None
        self.labels: list[str] | None = None

    def resolve(
        self,
        size: int,
        coordinates: np.ndarray | None = None,
        labels: list[str] | None = None,
    ) -> None:
        """Bind the axis to a dimension of `size` states.

        Args:
            size: Number of states of the dimension in a plot's data.
            coordinates: Coordinate of each state, if the dimension has
                coordinates.
            labels: Display text of each state, if the states are named
                (e.g. categories).

        Raises:
            ValueError: If another plot resolved it to a different size, or
                the initial state is out of range.
        """
        if self._size is not None and self._size != size:
            raise ValueError(
                f"{type(self).__name__} '{self.name}' is shared by data with "
                f"{self._size} and {size} states."
            )
        if not 0 <= self._state < size:
            raise ValueError(
                f"Initial state {self._state} of '{self.name}' is out of range."
            )
        self._size = size
        if self.coordinates is None and coordinates is not None:
            self.coordinates = np.asarray(coordinates, dtype=float).reshape(-1)
        if self.labels is None and labels is not None:
            self.labels = list(labels)

    @property
    def states(self) -> list[int]:
        if self._size is None:
            raise ValueError(f"'{self.name}' is not bound to any data yet.")
        return list(range(self._size))

    @property
    def name(self) -> str:
        if self._title is not None:
            return self._title
        title = title_of(self.variable_or_axes)
        if title is not None:
            return title
        return type(self.variable_or_axes).__name__.removesuffix("Axes").lower()

    def label(self, state: int) -> str:
        """Display text of `state`: its label, else its coordinate (4
        significant digits), else the index."""
        if self.labels is not None:
            return self.labels[state]
        if self.coordinates is None:
            return str(state)
        return f"{self.coordinates[state]:.4g}"


class SliderAxis(ControlAxis):
    """One state at a time, chosen with a slider."""


class TimeAxis(ControlAxis):
    """One state at a time, played as frames; at most one per figure.

    Args:
        variable_or_axes, init_state, title: As for :class:`ControlAxis`.
        duration: Mean display time per frame in milliseconds.

    Attributes:
        duration (int): Mean display time per frame in milliseconds.
    """

    def __init__(
        self,
        variable_or_axes: Variable | Axes | str,
        init_state: int = 0,
        title: str | None = None,
        duration: int = 500,
    ) -> None:
        super().__init__(variable_or_axes, init_state, title)
        self.duration = duration

    def frame_weights(self) -> list[float]:
        """Relative display time per frame, with mean 1: proportional to the
        time until the next frame if the states have coordinates, equal
        otherwise.

        Returns:
            list of float: One weight per state.
        """
        n = len(self.states)
        if self.coordinates is None or n < 2:
            return [1.0] * n
        steps = np.abs(np.diff(self.coordinates))
        if steps.sum() == 0:
            return [1.0] * n
        steps = np.append(steps, steps.mean())
        return list(steps / steps.mean())


class FacetAxis(ControlAxis):
    """Every state side by side: the overlay is repeated per state within
    its cell; the copies share the overlay's axes and scales.

    Args:
        variable_or_axes, init_state, title: As for :class:`ControlAxis`.
        orientation: ``"row"`` or ``"col"``: whether the copies are stacked
            or placed side by side.

    Raises:
        ValueError: If `orientation` is neither ``"row"`` nor ``"col"``.

    Attributes:
        orientation (str): ``"row"`` or ``"col"``.
    """

    def __init__(
        self,
        variable_or_axes: Variable | Axes | str,
        orientation: str = "col",
        title: str | None = None,
    ) -> None:
        if orientation not in ("row", "col"):
            raise ValueError(f"orientation must be 'row' or 'col', got {orientation!r}.")
        super().__init__(variable_or_axes, 0, title)
        self.orientation = orientation


class OverlayAxis(ControlAxis):
    """Every state at once in the same overlay, one drawn copy per state,
    each with a legend entry. The copies get categorical colors unless the
    plot binds a color."""

    copies = True

    @property
    def trained_states(self) -> list[int]:
        # Every copy is drawn in every evaluation.
        return [self.state]


class FixedAxis(ControlAxis):
    """One fixed state, named in the title of the cell.

    Raises:
        ValueError: When `state` is set to another state.
    """

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, value: int) -> None:
        if value != self._state:
            raise ValueError(f"FixedAxis '{self.name}' cannot change its state.")

    @property
    def trained_states(self) -> list[int]:
        return [self._state]


class Selector(Control):
    """Alternative references for one channel, chosen by the viewer; used
    in place of a reference, e.g. ``color=Selector([u, p])``. Only the
    chosen one is drawn, and titles derived from it follow the choice.

    Args:
        candidates: At least two Variables, or two names of table columns.
        init_state: Index of the initial choice.

    Raises:
        ValueError: With fewer than two candidates.

    Attributes:
        candidates (list of Variable or str): The candidates.
        selected (Variable or str): The chosen candidate.
    """

    def __init__(self, candidates: list[Variable] | list[str], init_state: int = 0) -> None:
        if len(candidates) < 2:
            raise ValueError("A Selector needs at least two candidates.")
        super().__init__(init_state)
        self.candidates = list(candidates)

    @property
    def selected(self) -> Variable | str:
        return self.candidates[self._state]

    @property
    def states(self) -> list[int]:
        return list(range(len(self.candidates)))

    @property
    def name(self) -> str:
        return " / ".join(self.label(state) for state in self.states)

    def label(self, state: int) -> str:
        """Display text of `state`: the candidate's title (math mode for a
        Variable)."""
        return title_of(self.candidates[state]) or ""
