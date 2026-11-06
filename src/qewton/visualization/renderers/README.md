# Renderers

A renderer draws one `Figure` in one plotting framework and one theme.
Plots and figures stay framework-neutral; everything in a framework's
vocabulary (its figure, styling, embedded controls, export) lives here.
See the [visualization README](../README.md).

## Available renderers (`__init__.py`)

`RENDERER_DICT` holds the renderers whose packages are installed, in
order of priority: `"plotly"` (`PlotlyRenderer`, if Plotly is installed),
then `"matplotlib"` (`MatplotlibRenderer`, always: Matplotlib is a core
dependency). `DEFAULT_RENDERER`, the renderer of a figure created without
one, is the first of them. A new renderer is one more guarded import and
entry; renderers are passed to a `Figure` as instances.

## Interface (`base.py`)

- **`Renderer(theme)`**: bound to one figure by `setup(figure)`, which
  builds the framework's figure with the cell grid. The figure calls
  `draw(key, result, cell, visible)` per drawn copy (creating or updating
  the artist of `key`), `hide(key)`, `draw_axes(cell)` and
  `draw_colorbar(scale, title, ticks, cells, visible)`. To embed a
  control it takes a `snapshot(keys, cells, scales)` per state and passes
  them to `animate`, `apply_slider` or `apply_selector`; the class
  attributes `embeddable` (kinds of control) and `max_embedded` (per
  output, None for no limit) declare what it embeds. Outputs: `show`,
  `save_html`, `save_image`, `save_gif`.
- **`Artist`**: draws one Result into one cell: `create(backend_figure,
  result, cell, theme)`, `update(...)` in place, `set_visible(...)`. An
  artist sees Results, cells and the theme, never plots. `Renderer.artists`
  maps (Result class, number of positions) to the artist class; a Result
  without one raises a ValueError naming the renderer.
- **`text.py`**: texts of the core (plain parts and `$...$` math) as
  parts or as plain text, for every renderer.

## Plotly (`plotly/`)

| Module | Holds |
|---|---|
| `__init__.py` | `PlotlyRenderer`: cell grid, axes, colorbars, snapshots and embedding, outputs |
| `primitives.py` | the artists of the primitive Results |
| `graph.py`, `card_geometry.py` | `NodeLinkArtist` for computation graphs and the geometry of its cards |
| `traces.py` | colorscales, value bins, segment and triangle coordinates, colorbar placement |
| `text.py` | texts with math as Plotly displays them |
| `camera.py` | the script moving the cameras of linked 3D cells together |

`PlotlyRenderer(theme=DEFAULT_THEME, mathjax=True)`: with MathJax, every
text naming a Variable is written as one math text, and `show()` in a
notebook displays HTML that loads plotly.js and MathJax from a CDN;
`mathjax=False` writes Plotly's own markup and needs no CDN. Embedded
controls become frames (time), slider steps and dropdown buttons that set
only what differs between states; animations start with their play
button and switch frames without transitions. Cells sharing position axes
zoom together; linked 3D cells move their cameras together in HTML
outputs and Dash.

```python
from qewton.visualization import Figure
from qewton.visualization.renderers import PlotlyRenderer

backend_figure = Figure(plot, renderer=PlotlyRenderer(mathjax=False)).draw()
```

## Matplotlib (`matplotlib/`)

| Module | Holds |
|---|---|
| `__init__.py` | `MatplotlibRenderer`: cell grid, axes, colorbars, snapshots and frames, outputs |
| `primitives.py` | the artists of the primitive Results, and `axes_of(backend_figure, cell)` |
| `style.py` | the theme as rcParams (fonts, sizes in points, colors), colormaps and norms |
| `text.py` | texts with math as mathtext displays them |

`MatplotlibRenderer(theme=DEFAULT_THEME, usetex=False, dpi=100)` draws a
`matplotlib.figure.Figure` without pyplot; `draw()` returns it for further
editing. Texts are mathtext in Computer Modern (text in the theme's serif
fonts, falling back to `cmr10`); math mathtext cannot parse is written as
plain text, with a warning; `usetex=True` renders all text with LaTeX and
needs `latex`.

| Result | 2D | 3D |
|---|---|---|
| `HeatmapResult` | `pcolormesh` | – |
| `SurfaceResult` | `tripcolor` (Gouraud) or one fill, `triplot` edges | `Poly3DCollection`, shaded if uniform |
| `LineResult` | `LineCollection` | `Line3DCollection` |
| `MarkerResult` | `scatter` | `scatter` |
| `BarResult` | `bar` | – |
| `ImageResult` | `pcolormesh` of RGB(A) cells | – |
| `ArrowResult` | `quiver` | `quiver` |

Not drawn: `ParallelCoordinatesResult` and `GraphLayoutResult`. 2D cells sharing position axes share their
Matplotlib axes; 3D cells are not linked. One TimeAxis or SliderAxis per
output is embedded as frames: `show()` and `save_html()` give
Matplotlib's JavaScript player, `save_gif()` a GIF with the frame weights
as durations; the figure title names the state of a frame. Images: PNG,
SVG and PDF without external tools. `show()` displays HTML in a notebook
and opens a window of Matplotlib's backend elsewhere.

```python
from qewton.visualization import Figure, WidgetApplication
from qewton.visualization.renderers import MatplotlibRenderer

figure = Figure(plot, renderer=MatplotlibRenderer())
figure.save_svg("figure.svg")
WidgetApplication.create(figure)  # in a notebook: a widget per control
```

See [DECISIONS.md](DECISIONS.md) for design decisions.
