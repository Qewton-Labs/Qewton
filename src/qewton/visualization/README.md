# Visualization

Plotting for Qewton data, tables, geometries and computation graphs. A
*plot* binds references into its input to the channels of one kind of
mark; *controls* reduce the dimensions it does not draw; *scales* map
values to positions, colors and arrow lengths. A *figure* arranges plots
in cells and draws them with a *renderer* (Plotly if installed, else
Matplotlib), which owns the theme. Requirements: [REQUIREMENTS.md](REQUIREMENTS.md).

## Plots and bindings

A data plot takes data, its `DataConfiguration` and bindings of channels
to references: Variables or Axes of the configuration.

- `position=` binds a geometry Variable or Axes (all its coordinates), a
  feature with up to three components, or a tuple of one-component
  references; it fills x, y, z in order. `x=`, `y=`, `z=` bind one
  component each and complement `position=`, e.g. `position=X, z=u`.
- Positions from the *domain* (coordinates of a geometry, index of a batch
  axis) draw that domain; positions from the *values* draw every point of
  the remaining dimensions. Every other dimension of the data must be
  reduced by a control, else construction raises `ValueError`.
- `color=` binds the color channel, `vector=` the arrows of a
  `VectorPlot`. A tuple in `y` or `z` draws one copy per reference.
- On a product geometry (e.g. a meshed space-time domain in one
  `GeometryAxes`), naming a factor in a binding or control splits it off.

| Plot | Draws |
|---|---|
| `HeatmapPlot` | a 2D domain flat, colored by a value |
| `SurfacePlot` | a surface in 3D: a 2D domain lifted by a height, a surface or the boundary of a volume |
| `LinePlot` | a line through a 1D domain, as a graph or a path |
| `ScatterPlot` | a marker per point (on the x axis for one position) |
| `BarPlot` | a bar per point of a 1D domain |
| `VectorPlot` | an arrow per point |
| `ImagePlot` | RGB(A) colors on a 2D grid |
| `TableScatter`, `TableHeatMap`, `ParallelCoordinatesPlot` | rows of a table (columns by name) |
| `GeometryPlot` | a geometry without data |
| `GraphPlot` | a computation graph |

See [plots/README.md](plots/README.md).

```python
from qewton.visualization import Figure, HeatmapPlot, SliderAxis

plot = HeatmapPlot(data, config, position=X, color=u, controls=[SliderAxis(t)])
Figure(plot).show()
```

## Controls and selectors

Control axes reduce one dimension (a batch axis, a one-dimensional
geometry or factor, or a table column) and differ only in presentation:

| Control | Shows |
|---|---|
| `SliderAxis` | one state, chosen with a slider |
| `TimeAxis` | one state at a time as animation frames (one per figure) |
| `FacetAxis` | every state side by side, one cell each |
| `OverlayAxis` | every state as a copy in one cell |
| `FixedAxis` | one fixed state, named in the cell title |

A `Selector` chooses which reference fills a channel, e.g.
`color=Selector([u, p])` or `x=Selector(["lr", "width"])` for table
columns. One control or Selector instance passed to several plots links
them. States are labeled by their coordinates or categories; animation
frames last in proportion to their time step.

## Scales

`Scale` maps values to positions, `ColorScale` to colors (`cmap`,
`reduce` for vectors), `VectorScale` to arrow lengths. Every scale is
trained over all states of the controls. Bounds are fixed with `vmin` and
`vmax`; `symmetric=True` centers on zero, `log=True` maps logarithmically
and labels by values. One instance shared by several plots gives one
range and one colorbar; a Selector gets one color scale per candidate
unless `color_scale=` gives a shared one or a list. Components of one
reference, e.g. the coordinates of a geometry, are drawn to one scale.
Drawn ranges end at the domain for rasters and surfaces and get a margin
for marks; bars start at zero. `categorical=True` (given to text columns
of a table) colors each category with the theme's palette, with a
colorbar step per category.

## auto_plot and auto_layout

`auto_plot(data, config)` chooses a plot type from the kind of value and
the drawn domain: a scalar over a 2D domain is a heatmap, over a 1D
domain a line, over a volume a surface; a vector with one component per
coordinate gives arrows; the coordinates of the domain give points;
several scalars become a Selector. Dimensions not drawn get
`default_control` (a slider); `controls=` sets them explicitly.
`plot_type=`, `value=`, bindings and plot options (e.g. `z=u`) take part
in the choice; if nothing fits, the error names every plot type tried.

`auto_layout(plots)` overlays plots that read well together (no two show
values the same way at the points of the domain, e.g. arrows over a
heatmap) and places the others side by side with linked axes.

## Figure and layout

`Figure(layout, renderer=None, title=None)` takes a plot, a list of plots
or nested `Overlay`/`Row`/`Column`. Plots in one `Overlay` share its
position axes (`PositionAxis(scale=, ticks=, title=)`); facets repeat a
cell.

| Output | Controls |
|---|---|
| `draw()` | current states; returns the renderer's figure |
| `show()`, `save_html()` | every control embedded (slider, animation, dropdown) |
| `save_png()`, `save_svg()` | current states |
| `save_gif()` | the animation of the `TimeAxis` |
| `DashApplication.create()` | sliders and selectors connected to widgets, the rest embedded |
| `WidgetApplication.create()` | sliders, animations and selectors connected to widgets |

Per plot, one control axis is embedded; next to it one Selector, else
several Selectors of different channels. Controls that would overwrite
each other, or that the renderer does not embed, keep their current
state, with a warning. Cells sharing position axes zoom, pan and rotate
together.

The size of a figure follows from its plots: every plot area gets the
theme's plot box (`plot_width` x `plot_height`), a fixed aspect fitted
into it; titles, tick labels, colorbars and legends get room beside it. A
colorbar stands right of the last cell of its scale, as tall as its plot
area; a legend, only where it tells two entries or more apart, right of
the last of the cells with these entries. Every output shows the figure at
its size; inline in a notebook, a wider figure scrolls.

## Renderers

`PlotlyRenderer(theme=DEFAULT_THEME, mathjax=True)` and
`MatplotlibRenderer(theme=DEFAULT_THEME, usetex=False, dpi=100)`; what
each draws and embeds: [renderers/README.md](renderers/README.md).

## Applications

`DashApplication.create(figure)` serves a Plotly figure as a Dash app;
start it with `app.run()`. `WidgetApplication.create(figure)` shows a
Matplotlib figure in a notebook (needs `ipywidgets` and `ipympl`).

## Themes

`Theme` sets colors, fonts and colormaps; `LIGHT_THEME` (default) and
`DARK_THEME` are given; `cmaps` holds colormap roles and colormaps. See
[themes/README.md](themes/README.md).

```python
from qewton.visualization import DARK_THEME, Figure
from qewton.visualization.renderers import PlotlyRenderer

Figure(plot, renderer=PlotlyRenderer(theme=DARK_THEME)).show()
```

## Graph integration

- `graph.visualize(port, ...)` evaluates a port on a mesh of its
  samplers' domains and returns a layout for `Figure`. With
  `reference=` (data with `reference_config`, another port, or a
  function of the points) it draws reference, prediction and `error=`
  with one color scale and shared controls; `variables=` narrows the
  output to some of its Variables.
- `data_node.visualize()` draws every output of a data node.
- `PlotNode` draws whatever data reaches it in a graph.

```python
layout = graph.visualize(model.output_ports[0], reference=exact, controls=[TimeAxis(t)])
Figure(layout).show()
```

See [DECISIONS.md](DECISIONS.md) for design decisions.
