# Themes

Colors, fonts and colormaps of Qewton figures.

- **`Theme`** (`base.py`): immutable dataclass with all visual settings:
  figure chrome (background, text, lines, grid), fonts, mark sizes,
  categorical palette, the colormap for each role, and node card colors
  and geometry for computation-graph plots. A renderer draws in its theme,
  e.g. `PlotlyRenderer(theme=DARK_THEME)`. `LIGHT_THEME` (default,
  `DEFAULT_THEME`) and
  `DARK_THEME` are predefined; custom themes override fields,
  e.g. `Theme(font_size_title=16)`.
- **`cmaps`** (`cmaps.py`, also `qewton.viz.cmaps`): colormaps
  for `ColorScale(cmap=...)`:
  - roles `cmaps.sequential`, `cmaps.diverging`, `cmaps.cyclic`, resolved
    against the figure's theme at draw time;
  - concrete colormaps such as `cmaps.meadow`, `cmaps.dusk`,
    `cmaps.redblue`, `cmaps.phase` and single-hue `cmaps.reds`,
    `cmaps.blues`, ..., the same in every theme;
  - any other value (a string such as `"Viridis"` or a colorscale list) is
    passed to the renderer unchanged.

  Without a cmap, a ColorScale uses the sequential role.
  `Theme.resolve_cmap()` performs this resolution.
- **`palette`** (`palette.py`): the raw color values - seven color
  families with four shades each, chrome tokens per theme, node card
  colors, the categorical order, and the colorscales built from them.

```python
from qewton.visualization import ColorScale, DARK_THEME, Figure, HeatmapPlot, cmaps
from qewton.visualization.renderers import PlotlyRenderer

# diverging: redblue in the light theme, orangeviolet_paper in the dark one
scale = ColorScale(cmap=cmaps.diverging)
plot = HeatmapPlot(data, config, position=X, color=u, color_scale=scale)
Figure(plot, renderer=PlotlyRenderer(theme=DARK_THEME)).show()
```

See [DECISIONS.md](DECISIONS.md) for design decisions.
