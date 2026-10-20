# Plots

Plot classes describe *what* to draw: which input is shown and how its
parts map onto visual roles. They do no drawing themselves. A `Figure`
evaluates each plot and hands the result to a renderer-specific artist
(see the [visualization README](../README.md)).

## Plot families

`Plot` (`base.py`) is the common base. Four input families derive from it:

- **`DataPlot`** (`data/`): `data` plus the `DataConfiguration`
  describing its axes. Grid plots (`HeatmapPlot`, `SurfacePlot`,
  `ImagePlot`, `EmbeddedGridPlot`, `QuiverPlot`), mesh plots
  (`MeshFieldPlot`, `MeshSurfacePlot`, `MeshVectorPlot`),
  `PointCloudPlot`, curves (`LinePlot`, `PathPlot`) and samples
  (`ScatterPlot`, `BarPlot`).
- **`TablePlot`** (`table/`): named columns, e.g. run logs or tuning
  results; categorical columns are coded to integers once.
  `ParallelCoordinatesPlot`, `TableScatter`, and `TableHeatMap`
  (`table/heatmap.py`, not exported by the package).
- **`GeometryPlot`** (`geometry/`): a geometry, drawn as its mesh.
- **`GraphPlot`** (`graph/`): a computation graph, drawn as a node-link
  diagram by `GraphLayout` (`graph/layout.py`).

Specs, controls, `Scale`, `Figure`, artists and the result contract are
shared by all families; only `evaluate()` and the control reduction differ.

## Specs (`spec.py`)

A spec maps one role of a plot onto a `Variable`/`Axes` (data plots) or a
column name (table plots). Plots accept plain Variables and wrap them in
the matching spec.

- **`AxisSpec`**: a structural axis (`x`, `y`, `z`), optionally
  logarithmic.
- **`ColorSpec`**: the quantity that colors the plot, with an optional
  colormap and `Scale`.
- **`VectorSpec`**: a 2D or 3D vector field, with arrow `scale`
  (None autoscales), `normalize` and `subsample`.
- **`Scale`**: a value range shared by every `ColorSpec` referencing the
  same instance; explicit `vmin`/`vmax` win over observed values, and
  `symmetric=True` centers the range on zero.
- **Controls** (`ControlSpec` subclasses), passed as `controls=[...]`, for
  dimensions that are not drawn directly: `SliderSpec` (interactive
  slider), `FixedSpec` (one fixed index), `FacetSpec` (one subplot per
  value), `TimeSpec` (animation frames). The owning plot resolves their
  defaults (bounds, values, physical coordinates) from the data.
- **`SelectorSpec`**: passed in place of a Variable to another spec,
  e.g. `ColorSpec(SelectorSpec([u, p]))`, to switch which candidate fills
  that role. It is found through `Plot.selector_specs`, not `controls`.

## From `evaluate()` to artists

1. `Figure.draw()` resets every `Scale`, collects each plot's
   `color_values()` (over all slider and time states) and trains the
   scales.
2. For each grid cell, `plot.create_artist(backend_figure, renderer,
   row, col)` picks the renderer's artist class (e.g.
   `renderer.HeatmapArtist`), which calls `plot.evaluate()`.
3. `evaluate()` reduces the input to the current control states (a data
   plot via `apply_controls()`, which also reduces geometry coordinates)
   and returns a renderer-neutral result dataclass from `result.py`:

   | Result                 | Produced by                                     |
   |------------------------|-------------------------------------------------|
   | `GridResult`           | `HeatmapPlot`, `SurfacePlot`, `ImagePlot`       |
   | `ParametricGridResult` | `EmbeddedGridPlot`, `TableHeatMap`              |
   | `MeshResult`           | mesh field/surface plots, `GeometryPlot`        |
   | `VectorResult`         | `QuiverPlot`, `MeshVectorPlot`                  |
   | `PointCloudResult`     | `PointCloudPlot`                                |
   | `CurveResult`          | `LinePlot`                                      |
   | `PathResult`           | `PathPlot`                                      |
   | `ScatterResult`        | `ScatterPlot`, `TableScatter`                   |
   | `TableResult`          | `ParallelCoordinatesPlot`                       |
   | `GraphLayoutResult`    | `GraphPlot`                                     |

4. The artist draws the result; when a control changes,
   `artist.update()` evaluates the plot again and refreshes its traces.

## Usage

```python
from qewton.visualization import (
    ColorSpec, Figure, HeatmapPlot, Scale, SliderSpec, cmaps,
)

shared = Scale(symmetric=True)
plot = HeatmapPlot(
    data, data_config, x=x, y=y,
    color=ColorSpec(u, cmap=cmaps.diverging, scale=shared),
    controls=[SliderSpec(t)],
)
Figure(plot).show()
```

Usually `auto_plot` chooses the plot type and roles; the classes here are
used directly to override that choice.

See [DECISIONS.md](DECISIONS.md) for design decisions.
