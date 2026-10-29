# Plots: design decisions

Decisions of the plots component (P in
[../REQUIREMENTS.md](../REQUIREMENTS.md)) that concern the subpackage only.

## One Result per drawn primitive (P1)

Results describe primitives (cells, triangles, segments, markers, bars,
arrows), not plot types: a heatmap on a mesh and a surface share
`SurfaceResult`, so a renderer needs one artist per primitive and
dimension, not per plot type.

## Plot types declare, the base class decides (P2)

What a plot type draws is declared as class attributes (`results`, `dims`,
`channels`, ...); binding checks, the result type and `supports()` are
derived from them in `DataPlot`. A new plot type declares and implements
`build()`; nothing else has to know it.

## Point encoding instead of a fill flag (P7)

A plot states how it shows values at the points of its domain (color,
arrows, or not at all), not only whether it fills an area: two arrow
fields hide each other as two heatmaps do, while arrows over a heatmap
read well.

## Table columns are not Variables (P6, C3)

Columns are named by strings, written as plain text, and chosen by
Selectors of names. A control on a column draws the rows of one value,
so a facet over a categorical column gives one cell per category. Log and
categorical columns are drawn by log10 and codes, with ticks given per
state, so switching the column switches the ticks.

## Parallel coordinates draw their own colorbar (P6, R1)

Parallel coordinates lay out their own axes, so they have no cell beside
which a colorbar of the figure could sit; their color scale is trained by
the figure but listed only under `scales`, and the artist draws the
colorbar.

## Graph layout without an external engine (GI4)

`GraphLayout` implements a layered layout (longest-path ranks plus
barycenter ordering) instead of graphviz, which would add a system
binary. Qewton graphs are mostly chains with a few composites, for which
this suffices. Node categories are an enum, assigned by an explicit table
of node classes.
