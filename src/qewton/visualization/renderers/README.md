# Renderers

Backends that turn a `Figure` and the evaluated results of its plots into
a concrete figure. Plots and `Figure` stay backend-neutral; everything in
a backend's vocabulary (subplot grid, theme styling, animation frames,
baked controls, export) lives here. Plotly is the only backend and the
default (`DEFAULT_RENDERER = PlotlyRenderer()`). For the overall picture
see [../README.md](../README.md).

## Public API

- **`Renderer`** (`base.py`): abstract backend interface.
  - `setup(figure)` builds an empty, themed backend figure sized to the
    figure's cell grid.
  - `reconcile_y_axis_titles(...)` applies Figure-level y-axis titles
    after all plots are drawn.
  - `animate(figure, backend_figure, spec)` adds one frame per
    `TimeSpec` value (called by `Figure.draw()`).
  - `apply_slider(...)` / `apply_selector(...)` bake a `SliderSpec` /
    `SelectorSpec` into a static figure (called by `Figure.show()` and
    `Figure.save_html()` only).
  - `show`, `save_html`, `save_gif`, `save_image` display or export.
- **`Artist`** (`base.py`): draws one plot (in one facet cell) into a
  backend figure.
  - `Artist.create(backend_figure, plot, row, col)` evaluates the plot,
    adds its traces and returns the artist.
  - `artist.update(backend_figure, plot, row, col)` re-evaluates the plot
    in its current control state and refreshes the same traces in place.
- **`PlotlyRenderer`** (`plotly/__init__.py`): the Plotly implementation.
  It also exposes every Plotly artist as a class attribute, so a plot's
  `create_artist(backend_figure, renderer, row, col)` calls e.g.
  `renderer.HeatmapArtist.create(...)` without importing artist modules.

## Plotly artists

| Module | Artists | Plots |
|---|---|---|
| `grid.py` | `ImageArtist`, `HeatmapArtist`, `ParametricHeatmapArtist`, `SurfaceArtist`, `ParametricSurfaceArtist` | `ImagePlot`, `HeatmapPlot`, `TableHeatMap`, `SurfacePlot`, `EmbeddedGridPlot` |
| `mesh.py` | `SurfaceMeshArtist`, `FilledMeshArtist`, `CurveFieldArtist` | `MeshFieldPlot` (3D / 2D / curve), `MeshSurfacePlot` |
| `vector.py` | `ArrowField2DArtist`, `ArrowField3DArtist` | `QuiverPlot`, `MeshVectorPlot` |
| `points.py` | `PointCloud2DArtist`, `PointCloud3DArtist` | `PointCloudPlot` |
| `tabular.py` | `ScatterArtist`, `BarArtist` | `ScatterPlot`, `TableScatter`, `BarPlot` |
| `table.py` | `ParallelCoordinatesArtist` | `ParallelCoordinatesPlot` |
| `graph.py` | `NodeLinkArtist` | `GraphPlot` |
| `curve.py`, `geometry.py` | `LineArtist`, `PathArtist`, `GeometryArtist`, `GeometryArtist2D` | `LinePlot`, `PathPlot`, `GeometryPlot` |

`common.py` holds the `PlotlyArtist` base class and shared helpers
(colorbar placement, shared `Scale` ranges, value binning, tensor
conversion). `card_geometry.py` is pure geometry for the beveled node
cards of `NodeLinkArtist`.

## How the pieces relate

1. `Figure.draw()` calls `renderer.setup(figure)` and then, per plot and
   facet cell, `plot.create_artist(...)` on the first draw or
   `artist.update(...)` on later draws.
2. After all plots, `reconcile_y_axis_titles()` runs, and with a
   `TimeSpec`, `animate()`.
3. `Figure.show()` / `save_html()` additionally call `apply_selector()`
   per `SelectorSpec` and `apply_slider()` for a single `SliderSpec`.
   `DashApplication` does not; its callbacks call `Figure.draw()` again,
   which runs `update()` with the new control states.

Every `PlotlyArtist` lists the traces it owns in `trace_indices`.
`animate()`, `apply_slider()` and `apply_selector()` set the control to
each state, replay `update()`, and capture exactly those traces (plus the
titles and types of their 2D axes) as frames, slider steps or dropdown
buttons. An artist that owns several traces must therefore list all of
them in `trace_indices` and keep their number fixed across updates.

## Usage

Renderers are used through `Figure`; they are rarely called directly.

```python
from qewton.visualization import Figure
from qewton.visualization.renderers.plotly import PlotlyRenderer

fig = Figure(plot, renderer=PlotlyRenderer())
backend_figure = fig.draw()  # a plotly.graph_objects.Figure
```

See [DECISIONS.md](DECISIONS.md) for design decisions.
