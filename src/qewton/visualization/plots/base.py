"""Base class :class:`Plot` shared by every plot family."""

from enum import Enum, auto

from qewton.visualization.axes import ControlAxis, Selector
from qewton.visualization.scales import ColorScale, Scale


class PointEncoding(Enum):
    """How a plot shows values at the points of its domain. Two plots of
    one encoding on one domain hide each other; plots of different ones
    read well together, e.g. arrows over a heatmap.

    Attributes:
        COLOR: By color, e.g. a heatmap, a surface, colored points of a
            domain or a filled geometry.
        ARROW: By arrows.
    """

    COLOR = auto()
    ARROW = auto()


class Plot:
    """Anything drawn into an Overlay.

    A plot evaluates, in the current states of its controls, to one Result
    per drawn copy (:meth:`evaluate`); the number of copies does not
    change with the states.

    Args:
        title: Heading of the plot's cell.
        label: Legend entry; None derives it from the plotted quantity.
        controls: Control axes reducing dimensions of the plot's input.

    Attributes:
        point_encoding (PointEncoding or None): How the plot shows values at
            the points of its domain; None if it shows them by position
            (e.g. a line graph or points in value space) or not at all.
            An Overlay draws plots encoding by color first. A class
            attribute, or a property where it depends on the plot.
        title (str or None): Heading of the plot's cell.
        label (str or None): The explicit legend entry.
        controls (list of ControlAxis): The control axes.
        selectors (list of Selector): Selectors in place of references.
        color_scales (list of ColorScale): The color scales the Figure
            draws colorbars of, trained over all states.
        active_color_scales (list of ColorScale): Those of `color_scales`
            shown in the current states (e.g. of the chosen Selector
            candidate).
        scales (list of Scale): Every scale of the plot besides the
            position axes, e.g. its color scales (also those without a
            colorbar of the Figure) and a VectorScale; the Figure resets
            them before training.
        color_index (int): Index of the first categorical color of the
            plot; set by the Figure so plots and their copies get distinct
            colors.
        dim (int): Number of position components drawn: 2 or 3, or 0 for a
            plot that lays out its own axes (e.g. parallel coordinates).
        n_copies (int): Number of Results :meth:`evaluate` returns.
        scaled_channels (tuple of str): Position channels drawn to one
            scale, e.g. the coordinates of a geometry, so it is not
            distorted; empty if none.
        color_title (str or None): Name of what is bound to the color
            channel in the current states; None if nothing named is.

    Notes:
        Subclasses implement `dim` and :meth:`evaluate`, and may override
        `selectors`, `color_scales`, `active_color_scales`, `scales`,
        `n_copies`, `scaled_channels`, `color_title`, :meth:`axis_title`,
        :meth:`axis_ticks`, :meth:`channels_of` and :meth:`train_scales`.
    """

    point_encoding: PointEncoding | None = None

    def __init__(
        self,
        title: str | None = None,
        label: str | None = None,
        controls: tuple[ControlAxis, ...] | list[ControlAxis] = (),
    ) -> None:
        self.title = title
        self.label = label
        self.controls = list(controls)
        self.color_index = 0

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def selectors(self) -> list[Selector]:
        return []

    @property
    def color_scales(self) -> list[ColorScale]:
        return []

    @property
    def active_color_scales(self) -> list[ColorScale]:
        return self.color_scales

    @property
    def scales(self) -> list[Scale]:
        return list(self.color_scales)

    @property
    def n_copies(self) -> int:
        return 1

    @property
    def scaled_channels(self) -> tuple[str, ...]:
        return ()

    @property
    def color_title(self) -> str | None:
        return None

    def axis_title(self, channel: str) -> str | None:
        """The title the plot gives a position axis.

        Args:
            channel: ``"x"``, ``"y"`` or ``"z"``.

        Returns:
            str or None: The name of what is bound to `channel`; None if
                nothing named is.
        """
        return None

    def channels_of(self, selector: Selector) -> tuple[str, ...]:
        """The channels a Selector of the plot fills.

        Args:
            selector: One of `selectors`.

        Returns:
            tuple of str: E.g. ``("color",)``.
        """
        return ()

    def axis_ticks(self, channel: str) -> tuple[list[float], list[str]] | None:
        """The ticks the plot gives a position axis in the current states,
        e.g. the categories of a categorical column.

        Args:
            channel: ``"x"``, ``"y"`` or ``"z"``.

        Returns:
            tuple or None: Tick positions and their labels; None for
                automatic ticks.
        """
        return None

    def train_scales(self, axes: dict[str, Scale]) -> None:
        """Train the plot's scales and the given position axis scales over
        all states of its controls. Restores the states afterwards.

        Args:
            axes: Scale of each position axis of the plot's overlay, by
                channel.
        """

    def evaluate(self) -> list:
        """Evaluate the plot in the current states of its controls.

        Returns:
            list of Result: One Result per drawn copy.
        """
        raise NotImplementedError
