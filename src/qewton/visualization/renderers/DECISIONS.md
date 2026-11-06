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
renderer on carrier traces that draw nothing else, one per color scale;
a cell laying out its own axes takes only domain traces, so its carrier
is a parallel coordinates trace with one hidden axis.

## Plotly: no transitions, no autoplay (R2)

Plotly would interpolate the coordinates of traces between frames, which
moves elements between the traces of different value bins. Frames switch
without transition, and an animation starts with its play button, since
playing on load would use Plotly's default transitions.

## Plotly: math only where Plotly renders it (R4)

Plotly renders math only through MathJax, only for a text that is math
as a whole, and never in slider labels, dropdown buttons or 3D scenes,
which are drawn by WebGL. With MathJax, a text naming a Variable is
written as one math text (plain parts as `\text`); slider labels,
buttons and scene axis titles get Plotly's markup (italics, sub- and
superscripts). Notebook frontends load no MathJax for Plotly's own
output, so `show()` displays HTML that loads it.

## Plotly: linked views (F5)

Cells sharing position axes are linked by Plotly's `matches`. Cells of
equal units are drawn to scale by their sizes. A cell with axes of its
own also keeps the scale on zooming (`scaleanchor`, shrinking the plot
area rather than widening a range); cells sharing an axis get no
`scaleanchor`, since Plotly then draws the ticks of the matched axes
displaced. 3D scenes have no linking of their own: a script in HTML
outputs and a clientside callback in Dash pass a camera change on to the
scenes of its group. Embedded states never set a camera or range, and `uirevision`
keeps the viewer's view when Dash redraws.

## Plotly: graph plots use shapes and a fixed pixel scale (GI4)

`NodeLinkArtist` draws cards, wires and labels as layout shapes and
annotations in the axes of its cell, with three invisible hover traces,
so the trace count does not grow with the graph. Its hidden axes, y
reversed (the layout grows downward), are its own; the figure's layout
sizes its cell. Cards are sized in data units but label offsets in
pixels, by `theme.graph_px_per_unit`.

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

## Plotly: a pixel layout without margins (F7, F8)

`make_subplots` spaces cells by fixed fractions that know nothing of tick
labels, titles or colorbars, and Plotly widens its margins for what it
draws outside the plot areas, shrinking the paper under places computed
for it (with ticks displaced from their axes). The figure is therefore
sized in pixels with fixed margins of 0, and every cell, title, colorbar,
legend and embedded control is placed by the renderer; a colorbar has no
padding within its length, so it is as tall as its plot area. A trace
laying out its own axes (parallel coordinates) gets its cell as domain.

## Plotly: legends set once (F9)

Artists draw no legend entry and updates keep the legend properties, so
`draw_legend` sets them once and embedding or an application's updates
leave them: all traces of a label share a `legendgroup` (a click toggles
every copy), the first trace of its first artist shows the entry.

## Plotly: scene cameras farther away (R8)

Plotly's default camera fits the box to the scene, so its tick labels and
titles are clipped at the edges; the camera is set farther once, when
the scene is created, so a viewer's rotation is kept.

## Matplotlib: colorbars inside their cell's Axes (F7, F8)

Colorbars of Matplotlib's layout are sized relative to their Axes, so they
change with the figure and miss a fixed aspect's plot area. A colorbar is
therefore an Axes inside its cell's, right of the plot area at a fixed
thickness: as tall as the plot area whatever its size; constrained layout
keeps room for it as for the Axes' labels. The figure is sized from the
estimated room, so drawing costs no rendering.

## Matplotlib: characters cmr10 lacks as math (T3)

Matplotlib's bundled Computer Modern (`cmr10`) is TeX-encoded: at a few
ASCII characters (`_ < > | { } \`) it holds other glyphs, which no font
fallback replaces. In plain text these are written as mathtext, whose
Computer Modern holds them.

