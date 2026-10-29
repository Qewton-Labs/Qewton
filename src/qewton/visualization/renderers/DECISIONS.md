# Renderers: design decisions

Decisions of the renderers component (R in
[../REQUIREMENTS.md](../REQUIREMENTS.md)) that concern the subpackage only.

## A renderer draws one figure (R1)

The renderer holds the framework's figure and the artists of one figure;
binding it to a second one raises. Keys given by the figure identify an
artist across states, so redrawing updates artists in place.

## Embedding writes what differs (R2, F3)

The figure redraws each state of a control; the renderer snapshots what
was drawn and writes only the properties that differ between the states.
Steps of different controls thereby leave each other's properties alone.
Frames of an animation hold the changed traces only; exporting them (GIF)
applies each frame to the whole figure.

## Colored 2D marks are binned into one trace per value bin (R1)

Plotly gives filled 2D scatters and lines one color per trace. Filled
triangles, colored lines and arrows are therefore grouped into a fixed
number of value bins, one trace each; the number of traces never changes
between states, so trace indices stay valid. Colorbars are drawn by the
renderer on invisible carrier traces, one per color scale.

## No transitions, no autoplay (R2)

Plotly would interpolate the coordinates of traces between frames, which
moves elements between the traces of different value bins. Frames switch
without transition, and an animation starts with its play button, since
playing on load would use Plotly's default transitions.

## Math only where Plotly renders it (R4)

Plotly renders math only through MathJax, only for a text that is math
as a whole, and never in slider labels or dropdown buttons. With MathJax,
a text naming a Variable is written as one math text (plain parts as
`\text`); slider labels and buttons get Plotly's markup (italics, sub-
and superscripts). Notebook frontends load no MathJax for Plotly's own
output, so `show()` displays HTML that loads it.

## Linked views (F5)

Cells sharing position axes are linked by Plotly's `matches`. Plotly
forbids an axis in both a `matches` and a `scaleanchor` constraint, so
only the first cell of a group is drawn to scale and the others follow
it. 3D scenes have no linking of their own: a script in HTML outputs and
a clientside callback in Dash pass a camera change on to the scenes of
its group. Embedded states never set a camera or range, and `uirevision`
keeps the viewer's view when Dash redraws.

## Graph plots use shapes and a fixed pixel scale (GI4)

`NodeLinkArtist` draws cards, wires and labels as layout shapes and
annotations in the axes of its cell, with three invisible hover traces,
so the trace count does not grow with the graph. Cards are sized in data
units but fonts in points, so a figure of one graph cell is sized from
`theme.graph_px_per_unit`.
