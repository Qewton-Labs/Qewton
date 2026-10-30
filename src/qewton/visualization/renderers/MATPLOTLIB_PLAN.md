# Matplotlib renderer: plan (temporary)

Implements [MATPLOTLIB_SPEC.md](MATPLOTLIB_SPEC.md). Requirement IDs refer to
the spec.

## 1. Module layout

```
visualization/
├── renderers/
│   ├── __init__.py          # hierarchy: RENDERER_DICT, DEFAULT_RENDERER   (changed)
│   ├── base.py              # + embeddable, max_embedded                   (changed)
│   ├── plotly/              # unchanged, except the two class attributes
│   └── matplotlib/          # new
│       ├── __init__.py      # MatplotlibRenderer
│       ├── primitives.py    # MatplotlibArtist and one artist per primitive
│       ├── style.py         # Theme -> rcParams, colormaps, norms
│       └── text.py          # labels as mathtext, plain text fallback
├── applications/
│   ├── __init__.py          # imports without dash/ipywidgets             (changed)
│   ├── base.py              # shared control logic moves here            (changed)
│   ├── dash_app.py          # lazy dash import                           (changed)
│   └── widget_app.py        # new: WidgetApplication
└── figure.py                # _embeddable reads the renderer's declaration (changed)
```

The subpackage is named `matplotlib`, like `plotly`. `import matplotlib`
inside it still resolves to the library, because imports are absolute.

## 2. Renderer hierarchy (R1–R6)

```python
# renderers/__init__.py
from .base import Renderer

#: Available renderers, in order of priority.
RENDERER_DICT: dict[str, type[Renderer]] = {}

try:
    from .plotly import PlotlyRenderer

    RENDERER_DICT["plotly"] = PlotlyRenderer
except ImportError:
    pass

from .matplotlib import MatplotlibRenderer  # core dependency: always available

RENDERER_DICT["matplotlib"] = MatplotlibRenderer

#: Renderer class of a Figure created without a renderer.
DEFAULT_RENDERER = next(iter(RENDERER_DICT.values()))
```

- `visualization/__init__.py` exports `MatplotlibRenderer` always, and
  `PlotlyRenderer` only if it is available. It uses the same `try` block.
- `RENDERER_DICT` only records priority and availability. There is no
  lookup by name (R6).

## 3. Embedding declaration (E1, E3)

```python
# renderers/base.py
class Renderer:
    #: Kinds of control a self-contained output can embed.
    embeddable: tuple[type, ...] = ()
    #: Most controls embedded per output; None for no limit.
    max_embedded: int | None = None

# PlotlyRenderer:     embeddable = (TimeAxis, SliderAxis, Selector); max_embedded = None
# MatplotlibRenderer: embeddable = (TimeAxis, SliderAxis);           max_embedded = 1
```

`Figure._embeddable` changes only in how it selects candidates:

```python
candidates = sorted(
    (c for c in requested if type(c) in _EMBEDDINGS), key=priority
)
for control in candidates:
    if (
        type(control) not in self.renderer.embeddable
        or len(accepted) == self.renderer.max_embedded
        or self._overwrites(control, accepted)
    ):
        skipped.append(control)
        continue
    ...
```

- The existing warning covers every skipped control.
- Its text becomes generic: "... only the current state is shown. Use a
  connected output (an application)."
- Plotly's behaviour stays the same.

## 4. MatplotlibRenderer

```python
class MatplotlibRenderer(Renderer):
    """Draws a Figure with Matplotlib, as a static figure; one TimeAxis or
    SliderAxis is embedded as frames."""

    artists = {
        (HeatmapResult, 2): HeatmapArtist,
        (SurfaceResult, 2): FlatMeshArtist,
        (SurfaceResult, 3): SurfaceArtist,
        (LineResult, 2): Line2DArtist,
        (LineResult, 3): Line3DArtist,
        (MarkerResult, 2): Marker2DArtist,
        (MarkerResult, 3): Marker3DArtist,
        (BarResult, 2): BarArtist,
        (ImageResult, 2): ImageArtist,
        (ArrowResult, 2): Arrow2DArtist,
        (ArrowResult, 3): Arrow3DArtist,
    }  # no ParallelCoordinatesResult, no GraphLayoutResult (out of scope)
    embeddable = (TimeAxis, SliderAxis)
    max_embedded = 1

    def __init__(self, theme: Theme = DEFAULT_THEME, usetex: bool = False, dpi: int = 100):
        # usetex: fail early (RuntimeError) if no `latex` executable is found.
        ...

    def style(self) -> contextlib.AbstractContextManager:
        """rc_context of the theme; wraps every public method (T4).
        Fonts and mathtext are resolved when the figure is drawn, so the
        exports run inside it as well."""

    # --- figure, axes, colorbars ---
    def create_backend_figure(self, figure) -> matplotlib.figure.Figure
        # Figure(layout="constrained"), not pyplot (M8); one subplot per cell
        # on a GridSpec; projection="3d" for dim 3; a cell sharing a
        # PositionAxis with an earlier cell gets sharex/sharey (M5).
    def draw(self, key, result, cell, visible=True) -> None
        # super().draw in style(); keeps the last (result, visible) per key
        # for snapshots.
    def draw_axes(self, cell) -> None
        # Builds and applies the axes settings: title, labels, limits, log scale,
        # ticks, aspect (M6); the legend when theme.show_legend and more
        # than one labelled artist.
    def draw_colorbar(self, scale, title, cells, visible=True) -> None
        # One fig.colorbar(ScalarMappable(norm, cmap), ax=[axes of cells])
        # per scale, kept by id(scale); update_normal() afterwards.

    # --- embedding (E2) ---
    def clear_embeddings(self) -> None          # drops self.animation
    def snapshot(self, keys, cells, scales) -> tuple[dict, dict, dict]
    def animate(self, time_axis, snapshots) -> None   # self.animation = FuncAnimation
    def apply_slider(self, slider, snapshots) -> None # the same, without autoplay
    def apply_selector(self, selector, snapshots)     # never called: not embeddable

    # --- outputs (O1–O4) ---
    def show(self) -> None        # notebook: IPython.display(HTML or PNG); else pyplot
    def html(self) -> str         # jshtml of the animation, else inline SVG
    def save_html(self, path) -> None
    def save_image(self, path, format, **kwargs) -> None   # savefig: png/svg/pdf
    def save_gif(self, path, fps=10, frame_weights=None) -> None
        # PillowWriter; a weight is realised by repeating frames.
```

### Cell → Axes

This applies to the Matplotlib renderer only: each renderer finds a cell's
subplot through its own backend API (Plotly uses `row`/`col` in
`add_trace`). The artist interface receives `(backend_figure, result, cell,
theme)`, and `backend_figure` stays a plain `matplotlib.figure.Figure`, so
`draw()` returns what users expect.

`create_backend_figure` gives each subplot Matplotlib's own `label`
property, e.g. `"cell-1-2"`. The Matplotlib artists find their Axes by this
label. Colorbar axes carry no such label.

```python
# renderers/matplotlib/primitives.py
def axes_of(backend_figure, cell) -> Axes:
    """The Axes of `cell`, labelled by create_backend_figure."""
    return next(ax for ax in backend_figure.axes if ax.get_label() == cell_label(cell))
```

### Snapshots

`Renderer.snapshot` already exists and returns a renderer-specific
`object`. Plotly returns a tuple of property dicts. Matplotlib does the
same and adds no new classes:

```python
snapshot(keys, cells, scales) -> (
    {key: (result, visible)},        # last drawn per artist key
    {(row, col): axes_settings},     # dict built and applied by draw_axes
    {id(scale): colorbar_settings},  # dict built and applied by draw_colorbar
)
```

`animate` and `apply_slider` build one `matplotlib.animation.FuncAnimation`
and keep it as `self.animation`. Its frame function re-applies a snapshot
to the existing artists through `artist.update`/`set_visible` and the two
settings dicts. `html()` uses `to_jshtml()`; `save_gif` uses
`PillowWriter`.

## 5. Artists (`primitives.py`, M2–M4, M7)

```python
class MatplotlibArtist(Artist):
    """Owns the Matplotlib objects it drew into one Axes."""

    def __init__(self, axes: Axes, objects: list) -> None: ...

    @classmethod
    def draw_into(cls, axes, result, theme) -> list:   # subclass: the objects
        raise NotImplementedError

    @classmethod
    def create(cls, backend_figure, result, cell, theme):
        axes = axes_of(backend_figure, cell)
        return cls(axes, cls.draw_into(axes, result, theme))

    def update(self, backend_figure, result, cell, theme):
        # Default: remove the objects and draw anew into the same Axes.
        # Subclasses whose geometry is fixed override with set_array /
        # set_offsets / set_segments.

    def set_visible(self, backend_figure, visible): ...
```

| Artist | Matplotlib | In-place update |
|---|---|---|
| HeatmapArtist | `pcolormesh` | `set_array` |
| FlatMeshArtist | `tripcolor(shading="gouraud")`, `triplot` for edges, `PolyCollection` for a fill without values | `set_array` |
| SurfaceArtist | `Poly3DCollection` with face colors from cmap/norm | redraw |
| Line2DArtist | `LineCollection` (per-segment colors) | `set_segments`, `set_array` |
| Line3DArtist | `Line3DCollection` | redraw |
| Marker2DArtist / Marker3DArtist | `scatter` | `set_offsets` / redraw |
| BarArtist | `bar` | redraw |
| ImageArtist | `imshow(extent=...)` | `set_data` |
| Arrow2DArtist | `quiver` (color by norm) | `set_UVC` |
| Arrow3DArtist | `Axes3D.quiver`, one color per group | redraw |

M4 already holds: `Renderer.artist` raises
`"MatplotlibRenderer cannot draw a ParallelCoordinatesResult in 0D."`

## 6. Theme (`style.py`, T1–T4)

```python
def rc_params(theme: Theme, usetex: bool) -> dict
    # facecolors, text/axes/grid colors, font sizes, axes.grid, spines
    # (axis_box), lines.linewidth, lines.markersize, axes.prop_cycle from
    # primary_color_palette, font.family="serif",
    # font.serif=[*split(theme.font_family), "CMU Serif", "cmr10"],
    # mathtext.fontset="cm", axes.formatter.use_mathtext=True,
    # text.usetex=usetex

@functools.cache
def colormap(resolved) -> matplotlib.colors.Colormap
    # Theme.resolve_cmap result: a stop list -> LinearSegmentedColormap,
    # a name -> matplotlib.colormaps[name] (T2)

def norm(scale: ColorScale) -> Normalize      # LogNorm for log scales
```

- `text.py`, the counterpart of `plotly/text.py`:

  ```python
  def as_mathtext(text: str | None) -> str | None:
      """`text` unchanged if mathtext parses it, e.g. "$t$ = 0.5"; otherwise
      its plain form (as_plain) and a UserWarning naming the text."""
  ```

  Checking with `matplotlib.mathtext.MathTextParser` is cached per text.
  The renderer passes titles, axis titles, legend labels and colorbar
  titles through `as_mathtext`. With `usetex=True` the text is passed as
  is.

## 7. Applications (A1–A4)

```python
# applications/base.py
class RenderApplication:
    """Base class for wrapping a Figure in an interactive application."""

    #: Kinds of control that get a widget; the rest is embedded.
    connected: tuple[type, ...] = ()

    @classmethod
    def renderer_classes(cls) -> tuple[type, ...]:
        """Renderers whose figures the application shows (imported lazily)."""
        raise NotImplementedError

    @classmethod
    def check_renderer(cls, figure) -> None:      # TypeError otherwise (A1)
    @classmethod
    def interactive_controls(cls, figure) -> list[Control]   # moved from Dash
    @classmethod
    def embedded_controls(cls, figure) -> list[Control]      # moved from Dash
    @classmethod
    def update(cls, figure, controls, states) -> object      # moved from Dash
    @classmethod
    def create(cls, figure):
        raise NotImplementedError


def require(*packages: str) -> None:
    """Raise an ImportError naming the missing packages (R5)."""
```

```python
# applications/dash_app.py: dash is imported inside create()
class DashApplication(RenderApplication):
    connected = (SliderAxis, Selector)          # TimeAxis stays embedded

    @classmethod
    def renderer_classes(cls):
        from qewton.visualization.renderers.plotly import PlotlyRenderer
        return (PlotlyRenderer,)
```

```python
# applications/widget_app.py
class WidgetApplication(RenderApplication):
    """Shows a Figure in a notebook: an ipympl canvas and one widget per
    control: an IntSlider per SliderAxis, a Play widget linked to a slider
    per TimeAxis, a Dropdown per Selector. A change sets the state, redraws
    the figure in place and refreshes the canvas."""

    connected = (SliderAxis, TimeAxis, Selector)  # nothing embedded

    @classmethod
    def renderer_classes(cls):
        from qewton.visualization.renderers.matplotlib import MatplotlibRenderer
        return (MatplotlibRenderer,)

    @classmethod
    def create(cls, figure) -> "ipywidgets.VBox":
        require("ipywidgets", "ipympl")
        # canvas from ipympl for the existing matplotlib Figure;
        # widgets observe "value" -> cls.update(...); canvas.draw_idle()
```

- The `TypeError` for a wrong renderer replaces Dash's current
  `ValueError`. The tests are adjusted accordingly.

## 8. Tests and CI (Q1–Q5)

| Change | Where |
|---|---|
| `"matplotlib": MatplotlibRenderer` in `RENDERERS` | `test_matrix`, `test_matrix_auto`, `test_matrix_tables`, `test_matrix_scales`, `test_matrix_layout`, `test_graphs/test_matrix_visualize` |
| `APPLICATIONS = {"plotly": DashApplication, "matplotlib": WidgetApplication}` for the connected output | the same matrices |
| expected-result entries `(plot, "matplotlib") → "not drawn: …"` for parallel coordinates and graphs | `test_matrix_tables`, `test_matrix_visualize` |
| expected warnings: more than one embeddable control, or a Selector, with Matplotlib | `test_matrix` embedding table |
| fallback: a subprocess with Plotly blocked by a meta-path hook → `DEFAULT_RENDERER is MatplotlibRenderer`, `import qewton` works | new `test_renderers.py` |
| renderer rules: rc not leaked, mathtext labels, LogNorm colorbar, shared axes, PNG/SVG/PDF, GIF frame count | new `test_matplotlib.py` |
| `MPLBACKEND=Agg` | `tests/conftest.py` |
| `pip install … ipywidgets ipympl` | `.github/workflows/main.yml` |
