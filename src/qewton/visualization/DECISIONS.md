# Visualization: design decisions

## auto_plot raises instead of guessing

`auto_plot` only picks a plot type where the axes determine it. Where a
different drawing is equally valid (e.g. BarPlot vs. LinePlot, PathPlot vs.
ScatterPlot, MeshSurfacePlot vs. MeshFieldPlot), or roles cannot be
assigned, it raises `ValueError` with an explanation. Leftover axes always
get a default control, because a Plot without controls for them fails at
`evaluate()`.

## Controls bind to whole 1D factors of a product domain

On a product geometry, a control can only step through a whole
1-dimensional factor, named by its Variable or a GeometryAxes holding only
that factor. Factors are split off by default only when the choice is
unambiguous; otherwise the caller has to name them.

## Shared spec instances link plots

Plots that share one control, `SelectorSpec` or `Scale` instance are
linked: one widget moves all of them, and a shared Scale gives one color
range with one colorbar after the last plot using it. Specs are compared
by identity for this.

## Figure is renderer-neutral

`Figure` computes the cell grid, cell dimensions, spans and titles; the
renderer translates them into its backend figure. Since a backend figure
fixes its grid when built, Figure rebuilds it whenever plots are added,
removed or replaced.

## Static exports bake at most one slider

`show()` and `save_html()` bake every state of a SliderSpec into the
figure with all other controls frozen. With several SliderSpecs most
combinations would be missing or wrong, so only the initial state is
shown and a warning points to `DashApplication`. SelectorSpecs are baked
the same way; Dash serves them live instead.

## One TimeSpec per Figure

Animation frames are figure-wide, so a Figure supports at most one
TimeSpec. The renderer builds the frames after the normal draw.
