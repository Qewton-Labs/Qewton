# Renderers: design decisions

## Artists own fixed traces and refresh them fully in update()

An artist creates its traces once in `create()` and `update()` rewrites
them in place from the plot's current control state. The number of traces
never changes between updates (binned artists keep empty bins), so trace
indices stay valid. Every trace an artist owns is listed in
`trace_indices`.

## Static controls are baked by replaying update()

`animate()`, `apply_slider()` and `apply_selector()` need no per-artist
code: they set the control to each state, replay `update()` and capture
the traces in `trace_indices` (and the titles/types of their 2D axes) as
Plotly frames, slider steps or dropdown buttons. Afterwards the control and
the live traces return to their initial state. The figure grows with the
number of states. Dash does not use this; it redraws live.

## Baked controls are limited where replay cannot be correct

Each baked state varies one control with all others frozen. Hence:

- A plot with more than one `SelectorSpec` raises `NotImplementedError`
  in `apply_selector()`; most combinations would silently show wrong data.
- A `TimeSpec` on a faceted plot raises `NotImplementedError` in
  `animate()`, and a `SliderSpec` on a faceted plot is skipped with a
  warning: `update()` has no per-cell facet state to replay.

`DashApplication` handles these cases.

## TimeSpec uses frames, sliders and selectors use update buttons

Only `animate()` uses Plotly frames (play/pause, frame slider, GIF
export). Sliders and dropdowns use `method="update"` steps, so they can
coexist with an animation on the same figure.

## Per-element colors are binned into one trace per value bin

Plotly's 2D `go.Scatter` fills and lines have one color per trace, and
`go.Cone` only colors by its own vector norm. Filled 2D meshes, 2D field
curves and arrow fields (2D and 3D) therefore group elements into
`plot.n_bins` value bins, one flat-colored trace each, plus an invisible
carrier trace for the continuous colorbar. Filled 2D meshes use
`go.Scatter` instead of `go.Mesh3d` at z = 0 so they stay in a cartesian
cell that can share a row with heatmaps. Cones use `sizemode="raw"` so
cone lengths stay comparable across bins.

## Graph plots draw with shapes and a fixed pixel scale

`NodeLinkArtist` draws cards, wires and labels as layout shapes and
annotations, with exactly three invisible hover traces, so the trace count
does not grow with the graph. Card sizes are in data units but fonts in
points, so an unsized, non-faceted graph figure gets a pixel size from
`theme.graph_px_per_unit`, and `show()` / `save_html()` disable Plotly's
responsive mode for figures with explicit size.
