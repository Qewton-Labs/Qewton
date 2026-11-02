# Spec: figure appearance

Figures of every renderer meet the conventions of scientific publications:
the data fill the figure, nothing overlaps, and every element that is no
data earns its place (Tufte's data-ink). The rules are renderer-neutral;
each renderer realizes them in its framework.

## Size and space

| ID | Requirement |
|---|---|
| L1 | The size of a figure follows from its grid: every cell gets a size from the theme, plus the space of its colorbars, legend and titles. |
| L2 | A cell with a fixed aspect ratio (equal position units) is sized to that ratio; no empty band is left beside its data. |
| L3 | A figure is shown at its own size and resolution in every output, inline in a notebook too; a figure wider than the output area scrolls horizontally instead of being scaled down. |
| L4 | Space between cells holds the axes titles, tick labels and colorbars of the cells next to it, so nothing overlaps. |

## Axes

| ID | Requirement |
|---|---|
| X1 | A raster or surface on a domain (heatmap, image, surface) ends exactly at its domain. |
| X2 | Markers, lines and arrows get a margin, so no mark is clipped at the edge; bars start at zero and get a margin at their tips. |
| X3 | Axis titles and tick labels are never clipped, in 3D scenes too; math a framework cannot set in a place (e.g. Plotly scenes) is written in its markup. |
| X4 | Numbers are written with the precision their range needs. |

## Colorbars and legends

| ID | Requirement |
|---|---|
| C1 | A colorbar stands right of the plot area of the last cell of its scale, as tall as that plot area, with its title beside it; it never covers a plot. |
| C2 | A legend appears only if it tells entries apart or toggles them (Plotly); a mark whose color comes from a colorbar has no entry, and equal entries appear once. |
| C3 | A legend stands outside the plot areas and next to no colorbar. |

## Categorical colors

| ID | Requirement |
|---|---|
| K1 | A categorical color scale maps each category to one color of a categorical colormap; the role `cmaps.categorical` resolves to the theme's palette, in the order plots are colored. |
| K2 | A categorical channel (e.g. a text column of a table) gets a categorical color scale without being asked. |
| K3 | A categorical colorbar shows one step per category, labeled by its name. |

## Style

| ID | Requirement |
|---|---|
| T1 | `LIGHT_THEME` and `DARK_THEME` keep their colors and TeX-like type and follow the data-ink rules: axis lines left and bottom with outward ticks, no frame, no or faint grid, legends without a box. Every rule is a theme field, so a theme can switch it. |
| T2 | Title, axis title and tick label sizes are consistent across cells and renderers. |
| T3 | Every text renders in the theme's fonts; characters a font lacks fall back to one that has them. |
| T4 | Arrows have heads in every renderer. |

## Out of scope

Graph plots (cards and wires) keep their layout; their colors come from the
theme as before.

## Acceptance

A gallery of typical figures (single and gridded heatmaps of square and
wide domains, shared and separate color scales, facets, overlays, lines,
bars, scatters, vectors, 3D surfaces, table plots with categorical and log
columns, `Graph.visualize`) shows none of the problems above with either
renderer; the rules that can be read from a drawn figure (sizes, ranges,
colorbar positions, legend entries, colorbar steps) are checked in the test
matrices.
