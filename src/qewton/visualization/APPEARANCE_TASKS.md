# Tasks: figure appearance

Order of work for [APPEARANCE_PLAN.md](APPEARANCE_PLAN.md). Every task ends
with the full test suite green and the gallery rendered with both
renderers.

| # | Task | Requirements | Done when |
|---|---|---|---|
| 1 | **Appearance matrix.** `test_matrix_appearance.py` with the gallery cases (single, wide and gridded heatmaps, shared and separate scales, facets, overlays, lines, bars, scatters, vectors, 3D surfaces, table plots, `Graph.visualize`) × renderer; tables of the expected properties, filled by the later tasks. | – | the cases draw with both renderers |
| 2 | **Cell geometry.** `Cell.aspect`, `Cell.box_aspect`; both renderers read them instead of `_aspect` / `_scene_aspect`. No visible change. | L2 | rule table for aspects; drawn figures unchanged |
| 3 | **Margins.** `Plot.margin`, `Overlay.margin`, `Cell.axis_range`; `BarPlot` trains 0; both renderers set ranges from `axis_range`, also for rasters. | X1, X2 | matrix: range exact or with margin per case |
| 4 | **Categorical colors.** `cmaps.categorical`, `Theme.resolve_cmap(..., steps=)`, `ColorScale.colorscale`, `color_range` ±0.5 for categorical scales; both renderers take colors only from `ColorScale.colorscale` (Matplotlib: `ListedColormap` for steps); `TablePlot` makes categorical color columns categorical. | K1–K3 | table matrix: one step per category in palette order, labeled by name, both renderers |
| 5 | **Theme.** `plot_width`, `plot_height`, `show_zero_line`, `legend_box`; data-ink defaults in `LIGHT_THEME` and `DARK_THEME` (no frame, no zero line, legend and colorbar without box); tick labels in `font_size_labels`, axis and cell titles in `font_size_axes`, in both renderers. | T1, T2 | rule tests of the theme settings per renderer |
| 6 | **Plot sizes.** `Figure.plot_sizes()` from aspects and the plot box. | L1, L2 | rule table: columns and rows per layout |
| 7 | **Plotly layout.** `plotly/layout.py` (`domains`, `text_width`); `create_backend_figure` sets size and domains, `constrain="domain"`; colorbars right of the plot area of their last cell, as tall as it, title beside; figure title centered. | L1, L2, L4, C1 | matrix: figure size, no colorbar over a plot area, colorbar height = plot height |
| 8 | **Matplotlib layout.** Size from `plot_sizes()`, grown by the measured gutters after constrained layout; colorbars as tall as their plot areas; title centered. | L1, L2, L4, C1 | same matrix checks for Matplotlib |
| 9 | **Legends.** `Figure.legends()`, `Renderer.draw_legend(labels, cells)` (base, recording renderer); Plotly: one legend per group with `legendgroup`, right of its last cell; Matplotlib: a figure legend per group. | C2, C3 | matrix: legends and entries per case; no legend over a plot area |
| 10 | **Notebook output and applications.** Both renderers show a figure in a notebook at its own size in a horizontally scrolling element; Plotly always through its HTML. `DashApplication` shows the figure at its size (no responsive resizing); `WidgetApplication` keeps the canvas at the figure's size in a scrolling box. Updates through widgets keep the layout. | L3 | HTML of `show()` holds the size and the scrolling element; connected outputs of the matrix keep size, colorbars and legends over all states |
| 11 | **3D.** Plotly scene axis titles in markup; scene domains with space for tick labels. | X3 | no `$` in scene titles; matrix checks for 3D cases |
| 12 | **Marks and texts.** Plotly arrow heads (fixed trace count); parallel coordinates with three significant digits; Matplotlib writes characters `cmr10` lacks as math. | T3, T4, X4 | rule tables; no `˙` for `_` |
| 13 | **Review.** Render the gallery with both renderers and both themes, compare with the spec, fix what remains. | all | no problem of the spec visible |
| 14 | **Docs and cleanup.** Requirements, READMEs and DECISIONS (visualization, renderers, themes) updated, each fact in its place; the three APPEARANCE files deleted. | – | docs reviewed |
