# Visualization

Plotting for Qewton data, geometries and computation graphs. Data is
described by a `DataConfiguration`; a *Plot* turns it into renderer-neutral
results, a *Figure* arranges plots and draws them through a *Renderer*
(Plotly by default), and a *Theme* sets colors and fonts.

## Pieces

- **Plots** (`plots/`): one class per kind of drawing. A Plot is built
  from data plus its `DataConfiguration` and *roles* (`x`, `y`, `color`,
  `vector`, ...) given as specs (`AxisSpec`, `ColorSpec`, `VectorSpec`).
  Families: data plots on grids, meshes, point clouds, curves and samples
  (`HeatmapPlot`, `MeshFieldPlot`, `QuiverPlot`, `LinePlot`, `ScatterPlot`,
  ...), `GeometryPlot`, `GraphPlot` for computation graphs, and table
  plots. `Plot.evaluate()` returns result objects (`GridResult`,
  `MeshResult`, ...) that renderers draw. See
  [plots/README.md](plots/README.md).
- **`auto_plot`** (`auto.py`): picks a Plot type and its roles from the
  axes of the `DataConfiguration` alone, adds default controls for
  leftover axes, and raises `ValueError` where the choice is ambiguous.
  With `plot_type=...` it only constructs that type.
- **Product domains** (`product_view.py`): data on a product geometry
  (e.g. a meshed space-time domain or `ProductSampler` output) is
  restructured before plotting. A factor whose Variable a control is bound
  to (e.g. `controls=[TimeSpec(T)]`) gets an axis of its own; the
  remaining factors are drawn as one geometry. If they would exceed three
  dimensions, or the plotted vector has fewer components, the
  1-dimensional factors get a default control when the choice is
  unambiguous.
- **Controls**: specs for axes that are not drawn directly. Where each one
  works:

  | Control        | Effect                                  | `Figure.show()` / `save_html()`        | `DashApplication`  |
  |----------------|-----------------------------------------|-----------------------------------------|--------------------|
  | `SliderSpec`   | steps through an axis                   | baked-in slider, only if it is the only SliderSpec; otherwise initial state and a warning | live slider |
  | `TimeSpec`     | animation over an axis                  | animation frames with play button; `save_gif()` | not interactive |
  | `SelectorSpec` | switches between quantities of a role   | baked-in dropdown                       | live dropdown      |
  | `FacetSpec`    | one grid cell per value (`row`/`col`)   | facet grid                              | facet grid         |
  | `FixedSpec`    | pins an axis to one state               | fixed                                   | fixed              |

  One control instance shared by several plots moves them together; the
  same holds for a shared `Scale` (one color range and colorbar).
- **Figure and layout** (`figure.py`, `layout.py`): `Figure` takes a Plot,
  a list of Plots (one row) or a nested `Overlay`/`Row`/`Column` layout.
  Panels are multiplied by facet grids. `draw()` returns the backend
  figure; `show()`, `save_html()`, `save_png()`, `save_svg()` and
  `save_gif()` export it.
- **Renderers** (`renderers/`): turn a Figure and its plots' results into
  a backend figure. `PlotlyRenderer` is the only one and the default. See
  [renderers/README.md](renderers/README.md).
- **Applications** (`applications/`): `DashApplication.create(figure)`
  serves a Figure as a Dash app with live sliders and dropdowns.
- **Themes** (`themes/`): `Theme` (`LIGHT_THEME` default, `DARK_THEME`)
  and the colormaps in `cmaps`. See [themes/README.md](themes/README.md).
- **`PlotNode`** (`nodes.py`): a graph node that plots whatever data
  reaches it via `auto_plot`.

## Usage

```python
from qewton.visualization import Figure, Row, TimeSpec, auto_plot

plot = auto_plot(u, data_config, controls=[TimeSpec(T)])
Figure(Row(plot, auto_plot(v, data_config))).show()
```

```python
from qewton.visualization import DashApplication, Figure, auto_plot

app = DashApplication.create(Figure(auto_plot(u, data_config)))
app.run()
```

See [DECISIONS.md](DECISIONS.md) for design decisions.
