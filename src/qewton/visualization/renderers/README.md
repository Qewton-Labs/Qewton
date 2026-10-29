# Renderers

A renderer draws one `Figure` in one plotting framework and one theme.
Plots and figures stay framework-neutral; everything in a framework's
vocabulary (its figure, styling, embedded controls, export) lives here.
`PlotlyRenderer` is the only renderer and the default
(`DEFAULT_RENDERER`). See the [visualization README](../README.md).

## Interface (`base.py`)

- **`Renderer(theme)`**: bound to one figure by `setup(figure)`, which
  builds the framework's figure with the cell grid. The figure calls
  `draw(key, result, cell, visible)` per drawn copy (creating or updating
  the artist of `key`), `hide(key)`, `draw_axes(cell)` and
  `draw_colorbar(scale, title, cells, visible)`. To embed a control it
  takes a `snapshot(keys, cells, scales)` per state and passes them to
  `animate`, `apply_slider` or `apply_selector`. Outputs: `show`,
  `save_html`, `save_image`, `save_gif`.
- **`Artist`**: draws one Result into one cell: `create(backend_figure,
  result, cell, theme)`, `update(...)` in place, `set_visible(...)`. An
  artist sees Results, cells and the theme, never plots. `Renderer.artists`
  maps (Result class, number of positions) to the artist class.

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

See [DECISIONS.md](DECISIONS.md) for design decisions.
