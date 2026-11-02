# Plan: figure appearance

Realizes [APPEARANCE_SPEC.md](APPEARANCE_SPEC.md). Renderer-neutral rules
(sizes, ranges, legend entries, categorical colors) are decided once in the
core and read by both renderers; each renderer only places them in its
framework.

## Causes found

| Problem | Cause |
|---|---|
| empty bands beside heatmaps (Plotly) | `scaleanchor` with the default `constrain="range"` widens the range to keep equal units in a cell of fixed size |
| tiny cells in grids (Plotly) | no `width`/`height`: every figure is 700×500 |
| colorbars over the next cell (Plotly) | `colorbar_position` centers the bar in `make_subplots`' fixed spacing, which knows nothing of bars, ticks and axis titles |
| colorbar as tall as the cell, not the plot (both) | the plot area is smaller than the cell box when the aspect is fixed |
| useless legends | every labelled Result gets an entry, also color-mapped ones and copies of facets |
| clipped marks, bars | ranges end at the data; bars do not train their baseline 0 |
| `$x_1$` in 3D (Plotly) | scenes get the MathJax text, which Plotly renders only in 2D |
| `_` as `˙` (Matplotlib) | the fallback font `cmr10` is TeX-encoded |
| arrows without heads (Plotly 2D) | arrows are drawn as plain segments |

## Core (renderer-neutral)

### Cell geometry: `figure.py`

The aspect logic, today duplicated in `_aspect` (Matplotlib) and
`_scene_aspect` (Plotly), moves to `Cell`; the figure sizes the plot areas
from it.

```python
class Cell:
    @property
    def aspect(self) -> float | None:
        """Height over width of the plot area of a 2D cell whose x and y
        are drawn to one scale (their spans), else None (free)."""

    @property
    def box_aspect(self) -> tuple[float, float, float] | None:
        """Box of a 3D cell (as the two renderers compute it today)."""

    def axis_range(self, channel: str) -> tuple[float, float] | None:
        """The drawn range of a position axis, in its transformed values:
        the scale's range, widened by Overlay.margin (X1, X2)."""


class Figure:
    def plot_sizes(self) -> tuple[list[float], list[float]]:
        """Width of every grid column and height of every row of plot
        areas, in pixels (L1, L2): a cell fits its aspect into the
        theme's plot box (theme.plot_width x theme.plot_height); a column
        is as wide as its widest cell, a row as tall as its tallest."""

    def legends(self) -> list[tuple[list[str], list[Cell]]]:
        """The legends (C2): the entries of every cell (labels of drawn
        Results without a color scale, each once, in drawing order), cells
        with equal entries grouped, as cells sharing a color scale share
        a colorbar; groups of fewer than two entries get none."""
```

`Figure.draw` passes each legend to a new renderer call
`draw_legend(labels, cells)` after the cells; it stands right of the last
of its cells, like a colorbar. Facets and comparison plots thereby share
one legend; cells of different content keep their own.

### Margins: `plots/base.py`, `layout.py`

```python
class Plot:
    #: Fraction of the span added on both sides of the drawn ranges; 0
    #: for plots ending at their domain (heatmap, image, surface).
    margin: float = 0.04

class Overlay:
    @property
    def margin(self) -> float:
        """0 if any of its plots ends at its domain, else the largest."""
```

`BarPlot` trains its y scale with 0, so bars start inside the range.

### Categorical colors: `themes/cmaps.py`, `themes/base.py`, `scales.py`

```python
# cmaps.py
categorical = ColormapRole("categorical")  # resolves to the palette

# base.py
class Theme:
    def resolve_cmap(self, cmap=None, steps: int | None = None):
        """... `categorical` resolves to primary_color_palette, in the
        order plots are colored; with `steps`, any colormap becomes
        `steps` equal bands (a palette: its first colors, a continuous
        one: sampled at the band centers)."""

# scales.py
class ColorScale:
    def colorscale(self, theme):
        """The resolved colormap of this scale: `cmap` (default: the
        categorical role for a categorical scale, else sequential),
        stepped once per category for a categorical scale. The one place
        both renderers take colors from."""

def color_range(scale):
    """... a categorical scale spans its codes +-0.5, so every band is
    centered on its code (K3)."""
```

`TablePlot` creates `ColorScale(categorical=True)` for a categorical color
column (K2); its colorbar ticks are the category names, as today.

### Theme: `themes/base.py`

New fields: `plot_width`, `plot_height` (the box a plot area fits into, px). Changed defaults
for `LIGHT_THEME` and `DARK_THEME`, colors and fonts unchanged (T1):
`axis_box=False`, `show_zero_line=False` (new), `legend_box=False` (new),
colorbars without an outline; tick labels in `font_size_labels`, axis and
cell titles in `font_size_axes` (T2).

## Plotly

Pixel layout instead of `make_subplots`' spacing. A new module
`plotly/layout.py` turns the plot sizes into paper coordinates:

```python
def domains(figure, theme) -> tuple[dict, tuple[int, int]]:
    """Paper domain of every cell, colorbar and legend, and the figure
    size in pixels: the plot sizes plus, per grid column, the space left
    (y tick labels and title) and right (colorbars and legends of its
    cells), per row above (cell title) and below (x tick labels and
    title)."""

def text_width(text, size) -> float:
    """Estimated width of a text in pixels: its visible characters (math
    as plain text) times the font size, with a small reserve."""
```

Tick labels are estimated from the longer formatted end of the range;
Plotly keeps choosing its ticks, so zooming relabels them.

- `create_backend_figure` sets `width`/`height` and the axis domains;
  `constrain="domain"` keeps equal units on zoom without widening ranges.
- `draw_colorbar`: x right of the plot area of the last cell of its scale,
  `len` its height, title `side="right"`; a stepped colorscale for
  categorical scales comes from `ColorScale.colorscale`.
- `draw_legend`: one Plotly legend (`legend`, `legend2`, ...) per legend
  of the figure, right of its last cell; entries grouped by label
  (`legendgroup`) and shown once, so a click toggles every copy.
- 3D axis titles use Plotly's markup (`plotly/text.py`), scenes get a
  domain with gutters for their tick labels (X3).
- `Arrow2DArtist`: two head strokes per arrow in the same traces, so the
  trace count stays fixed (T4).
- Parallel coordinates: range labels with three significant digits (X4).
- `show()` in a notebook always writes the HTML, wrapped in a
  horizontally scrolling element at the figure's own size (L3).

## Matplotlib

- Size: the figure starts at `plot_sizes()`; after constrained layout
  has placed everything, it grows by the difference between the drawn and
  the wanted plot areas, so the gutters are measured, not estimated (C1,
  L2). `_aspect` is replaced by `Cell`.
- Ranges from `Cell.axis_range`, also for rasters (no automatic margins).
- `draw_legend`: a figure legend right of the last of its cells (C3).
- `html()`: `<img>` at its own width with `max-width: none` in a
  horizontally scrolling element (L3).
- `text.py`: characters `cmr10` lacks (`_ { } < > | \`) are written as
  math, e.g. `$\_$` (T3).
- `style.colormap`: stepped stops become a `ListedColormap`.

## Tests (matrices)

- `test_matrix_appearance.py`: gallery case × renderer. Per case expected
  plot sizes, range kind (exact / margin), legend entries and colorbar
  cells; checks from the drawn figure that colorbars and legends overlap
  no plot area, colorbars are as tall as their plot areas and figures as
  large as their grid.
- Categorical colors in `test_matrix_tables.py`: steps and colors of the
  categorical colorbar for both renderers.
- Rules of a single place (aspect, margin, stepped colormaps, cmr10
  characters) as tables in the existing rule tests.

## Decisions

1. Legends are shared by cells of equal entries, placed like colorbars.
2. The plot box (`plot_width` × `plot_height`, default 400×320 px) bounds a plot area; fixed aspects
   fit into it, so the figure size follows the content.
3. A figure title stands centered over all plot areas, in both renderers.
4. Gutters: Plotly estimates them from character counts and font sizes;
   Matplotlib measures them in its own layout.
5. The plot area size is the same in every output; images change their
   resolution (`scale`, `dpi`), not their layout.
