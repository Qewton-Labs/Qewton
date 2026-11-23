"""Interactive notebook application for a Figure, with ipywidgets."""

import functools

from qewton.visualization.applications.base import RenderApplication, require
from qewton.visualization.axes import Control, Selector, SliderAxis, TimeAxis
from qewton.visualization.renderers.matplotlib import MatplotlibRenderer
from qewton.visualization.renderers.text import as_plain

# ipywidgets and ipympl are imported on first use, not with this module:
# importing ipympl after Matplotlib switches Matplotlib's global backend (see
# _ipympl_classes), and both are slow to import.


@functools.cache
def _ipympl_classes() -> tuple[type, type]:
    """The themed ipympl canvas class and ipympl's FigureManager.

    Importing ipympl when Matplotlib is already imported calls
    ``matplotlib.use("module://ipympl.backend_nbagg")``: the user's backend
    would be replaced (e.g. ``plt.show()`` in a script no longer opens a
    window). The application draws on a canvas of its own and needs no
    global backend, so that call is suppressed during the import.
    """
    # pylint: disable=import-outside-toplevel
    import matplotlib

    use = matplotlib.use
    matplotlib.use = lambda *args, **kwargs: None
    try:
        from ipympl.backend_nbagg import Canvas, FigureManager
    finally:
        matplotlib.use = use

    class _ThemedCanvas(Canvas):
        # An ipympl canvas that renders in the renderer's style: Matplotlib
        # reads part of it (e.g. ticks made on zooming) when rendering,
        # which the frontend triggers outside any call of the renderer.

        def __init__(self, figure, style) -> None:
            super().__init__(figure)
            self._style = style

        def draw(self) -> None:
            with self._style():
                super().draw()

    return _ThemedCanvas, FigureManager


class WidgetApplication(RenderApplication):
    """Shows a Figure in a notebook: an interactive Matplotlib canvas
    (ipympl: zoom, pan, 3D rotation) and a widget per control. A slider
    sets a SliderAxis, a play button with a slider a TimeAxis, a dropdown a
    Selector; a change sets the state and redraws the figure in place.
    Every such control is connected, so nothing is embedded.

    Example:
        >>> WidgetApplication.create(Figure(plot, renderer=MatplotlibRenderer()))

    See Also:
        :class:`~qewton.visualization.applications.base.RenderApplication`
    """

    connected = (SliderAxis, TimeAxis, Selector)

    @classmethod
    def renderer_classes(cls) -> tuple[type, ...]:
        """See :meth:`RenderApplication.renderer_classes
        <qewton.visualization.applications.base.RenderApplication.renderer_classes>`."""
        return (MatplotlibRenderer,)

    @classmethod
    def create(cls, figure):
        """Build the widgets displaying `figure`.

        Args:
            figure (Figure): The figure to show.

        Returns:
            ipywidgets.VBox: The canvas, at the figure's size in a box
                scrolling where it is wider than the output, above one row
                per control; display it as the last expression of a
                notebook cell.

        Raises:
            ImportError: If ipywidgets or ipympl is not installed.
            TypeError: If the figure is not drawn by a MatplotlibRenderer.
        """
        require(cls.__name__, "ipywidgets", "ipympl")
        import ipywidgets as widgets  # pylint: disable=import-outside-toplevel

        cls.check_renderer(figure)
        themed_canvas, figure_manager = _ipympl_classes()
        controls = cls.interactive_controls(figure)
        backend_figure = figure.draw(embedded=cls.embedded_controls(figure))
        canvas = themed_canvas(backend_figure, figure.renderer.style)
        figure_manager(canvas, id(figure))
        rows, values = [], []
        for control in controls:
            row, value = cls.widget(control)
            rows.append(row)
            values.append(value)

        def update(_change):
            cls.update(figure, controls, [v.value for v in values])
            canvas.draw_idle()

        for value in values:
            value.observe(update, names="value")
        # At the figure's own size, scrolling where it is wider.
        box = widgets.Box([canvas], layout=widgets.Layout(overflow="auto"))
        return widgets.VBox([box, *rows])

    @staticmethod
    def widget(control: Control) -> tuple:
        """The row of widgets setting a control: its name as math, then a
        slider (with a play button for a TimeAxis) or a dropdown.

        Args:
            control: The control.

        Returns:
            tuple: The row (``ipywidgets.HBox``) and the widget whose
                ``value`` is the state.
        """
        import ipywidgets as widgets  # pylint: disable=import-outside-toplevel

        options =[(as_plain(control.label(s)), s) for s in control.states]
        if isinstance(control, Selector):
            value = widgets.Dropdown(options=options, value=control.state)
            return widgets.HBox([widgets.HTMLMath(control.name), value]), value
        value = widgets.SelectionSlider(options=options, value=control.state)
        parts = [widgets.HTMLMath(control.name), value]
        if isinstance(control, TimeAxis):
            play = widgets.Play(
                min=0, max=len(options) - 1, value=value.index, interval=control.duration
            )
            widgets.link((play, "value"), (value, "index"))
            parts.insert(1, play)
        return widgets.HBox(parts), value
