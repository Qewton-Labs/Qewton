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

## Plotly: colored 2D marks are binned into one trace per value bin (R1)

Plotly gives filled 2D scatters and lines one color per trace. Filled
triangles, colored lines and arrows are therefore grouped into a fixed
number of value bins, one trace each; the number of traces never changes
between states, so trace indices stay valid. Colorbars are drawn by the
renderer on invisible carrier traces, one per color scale.

## Plotly: no transitions, no autoplay (R2)

Plotly would interpolate the coordinates of traces between frames, which
moves elements between the traces of different value bins. Frames switch
without transition, and an animation starts with its play button, since
playing on load would use Plotly's default transitions.

## Plotly: math only where Plotly renders it (R4)

Plotly renders math only through MathJax, only for a text that is math
as a whole, and never in slider labels or dropdown buttons. With MathJax,
a text naming a Variable is written as one math text (plain parts as
`\text`); slider labels and buttons get Plotly's markup (italics, sub-
and superscripts). Notebook frontends load no MathJax for Plotly's own
output, so `show()` displays HTML that loads it.

## Plotly: linked views (F5)

Cells sharing position axes are linked by Plotly's `matches`. Plotly
forbids an axis in both a `matches` and a `scaleanchor` constraint, so
only the first cell of a group is drawn to scale and the others follow
it. 3D scenes have no linking of their own: a script in HTML outputs and
a clientside callback in Dash pass a camera change on to the scenes of
its group. Embedded states never set a camera or range, and `uirevision`
keeps the viewer's view when Dash redraws.

## Plotly: graph plots use shapes and a fixed pixel scale (GI4)

`NodeLinkArtist` draws cards, wires and labels as layout shapes and
annotations in the axes of its cell, with three invisible hover traces,
so the trace count does not grow with the graph. Cards are sized in data
units but fonts in points, so a figure of one graph cell is sized from
`theme.graph_px_per_unit`.

## Renderers declare what they embed (R6, F3)

Frameworks embed controls differently: Plotly writes several sliders,
animations and dropdowns into one figure, Matplotlib's player and GIFs
have one axis of frames. Each renderer therefore declares the kinds of
control it embeds and how many per output, and the figure skips the
others with its usual warning, so no figure code depends on a renderer.

## Colorbar ticks come from the figure (F6)

Categories and log scales label their colorbar by names and values at
positions in the scale's transformed values. The figure computes these
ticks (from the plots, else from a log scale) and passes them to
`draw_colorbar`, so every renderer labels them alike.

## Matplotlib: themed per figure, no pyplot (R1, T1)

Matplotlib reads most settings from global rcParams, partly only when
rendering. The renderer builds figures with `matplotlib.figure.Figure`
rather than pyplot, and runs every drawing and export in a `rc_context`
of its theme; the widget application's canvas renders in that context,
too, since the frontend triggers rendering (e.g. on zooming).

## Matplotlib: mathtext by default (R4)

Mathtext renders the core's texts, plain and math mixed, without a LaTeX
installation; `usetex` is opt-in. It knows a subset of LaTeX and fails
only when rendering, so texts are checked when drawn.

## Matplotlib: native primitives only (R7)

Every primitive maps to one native Matplotlib primitive with continuous
colors, so no value bins are needed; what has none is left out. An
update draws an artist's objects anew at its place in the drawing order,
so overlays keep their order.

## Matplotlib: frames from snapshots (R2, R6)

A frame applies a snapshot to the existing artists, so frames and the
static figure are drawn by the same code. The animation is built only
when an output is written, and the current state is restored afterwards.
The player has no labels of its own, hence the title.
