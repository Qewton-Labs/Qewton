# Plots: design decisions

## Input families share everything below `evaluate()`

Data, table, geometry and graph plots are separate subclasses of `Plot`
because their inputs differ (a `DataConfiguration` describes tensor axes,
which tables, geometries and graphs do not have). Specs, controls,
`Scale`, `Figure` and artists are shared, so a new input family only
implements `evaluate()`, `create_artist()` and its own control reduction.

## `evaluate()` returns renderer-neutral results

Plots return plain dataclasses (`result.py`) and make no drawing
decisions; artists do all drawing. A new renderer only needs artists for
the result types. Results of colorable plots have a `color` field, so
`Plot.color_values()` trains a shared `Scale` for every family without
per-plot code.

## Shared color scales cover all control states

`color_values()` evaluates a plot over every `SliderSpec` and `TimeSpec`
state, so the color range stays fixed while scrubbing or playing instead
of jumping between states. Without a shared range, plots with different
value ranges would look identical side by side.

## Vector autoscale uses the whole data

With `VectorSpec(scale=None)`, the longest vector over all control states
is drawn slightly shorter than the typical arrow spacing. Arrow lengths
thus stay comparable while a slider moves, and neighboring arrows do not
overlap. Subsampling and NaN dropping only choose which arrows are
drawn; they never change what a drawn arrow encodes.

## Controls reduce data and geometry coordinates together

A control on a geometry dimension slices the data and the geometry's
coordinates (`reduce_coordinates()`) in the same way, so drawn positions
always match drawn values.

## `SelectorSpec` wraps a role, not a dimension

A selector switches which quantity fills a role, so it is passed in place
of a Variable and unwrapped by `PlotSpec.variable_or_axes`. All consumers
read the selected candidate without knowing about selectors. It is never
part of `controls`, because it does not reduce any dimension.

## `title` and `label` are separate

`title` is the panel heading, `label` the legend entry. They are kept
apart so one plot's title cannot appear as another plot's legend entry
in an overlay.

## Graph layout without an external engine

`GraphLayout` implements a simple layered layout (longest-path ranks plus
barycenter ordering) instead of using graphviz, which would add a system
binary dependency. Qewton graphs are mostly linear chains with a few
composites, for which this is sufficient.
