# Plots

Plot classes describe *what* is drawn: they bind references into their
input to channels and evaluate to renderer-neutral Results, one per drawn
copy. They never draw; a renderer's artists draw the Results (see the
[visualization README](../README.md)).

## Plot interface (`base.py`)

`Plot` is the base of every family. A Figure uses only this interface:
`evaluate()`, `train_scales(axes)`, `dim` (number of positions; 0 for a
plot that lays out its own axes), `controls`, `selectors`,
`channels_of(selector)`, `color_scales` (with a colorbar of the figure),
`active_color_scales`, `scales` (all to reset before training),
`n_copies`, `scaled_channels`, `point_encoding`, `axis_title(channel)`,
`axis_ticks(channel)` and `color_title`.

## Families

- **`DataPlot`** (`data/`): data plus its `DataConfiguration`. A type
  declares `channels`, `required` channels, `results` per domain
  (structure and topological dimension), the numbers of positions `dims`,
  whether positions must come from the domain, and whether colors are
  mapped through a ColorScale; it implements `build(positions, color,
  features)`. `supports(domain, n_positions)` answers from these
  declarations. Types: `HeatmapPlot`, `SurfacePlot`, `LinePlot`,
  `ScatterPlot`, `BarPlot`, `VectorPlot`, `ImagePlot`.
- **`TablePlot`** (`table/`): a table of named columns (a dict or a
  DataFrame); text and boolean columns are categorical. Channels take
  column names or Selectors of them; controls reference a column and draw
  the rows of one of its values; log columns are drawn by their log10
  with ticks labeled by values. Types: `TableScatter`, `TableHeatMap`,
  `ParallelCoordinatesPlot`.
- **`GeometryPlot`** (`geometry/`): a geometry without data: points,
  curves, surfaces with edges, volumes by their boundary.
- **`GraphPlot`** (`graph/`): a computation graph as a node-link diagram,
  laid out by `GraphLayout`.

## Results (`result.py`)

| Result | Drawn as | Produced by |
|---|---|---|
| `HeatmapResult` | colored cells of a rectilinear grid | `HeatmapPlot`, `TableHeatMap` |
| `SurfaceResult` | triangles, flat in 2D or a surface in 3D | `HeatmapPlot` (mesh), `SurfacePlot`, `GeometryPlot` |
| `LineResult` | segments through points | `LinePlot`, `GeometryPlot` |
| `MarkerResult` | a marker per point | `ScatterPlot`, `TableScatter`, `GeometryPlot` |
| `BarResult` | a bar per point | `BarPlot` |
| `ArrowResult` | an arrow per point | `VectorPlot` |
| `ImageResult` | colors on a uniform grid | `ImagePlot` |
| `ParallelCoordinatesResult` | a line per row through one axis per column | `ParallelCoordinatesPlot` |
| `GraphLayoutResult` | node cards, ports, edges, clusters | `GraphPlot` |

Every Result carries its legend entry, the index of its categorical
color, its color scale and title, and its opacity.

`color_channel.py` holds the color scales of a color binding, shared by
data and table plots.

See [DECISIONS.md](DECISIONS.md) for design decisions.
