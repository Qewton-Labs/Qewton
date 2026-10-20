# Themes: design decisions

## Colormap roles vs. concrete colormaps vs. renderer strings

A ColorSpec's `cmap` is one of three kinds. A role (`cmaps.sequential`,
`diverging`, `cyclic`) means "this kind of colormap, as the theme defines
it" and is resolved at draw time, so changing the theme changes every plot
that used a role. A concrete colormap (`cmaps.meadow`, ...) is fixed and
theme-independent. Strings and colorscale lists are never interpreted by
Qewton and go to the renderer as-is. A theme's `*_cmap` fields must be
concrete (or renderer-native), never roles, so resolution cannot loop.

## Colorscales are interpolated in OKLab

Colorscales are interpolated in OKLab between fixed palette anchors and
sampled at 64 stops. This keeps them perceptually even, which linear sRGB
interpolation between few stops does not; 64 stops keep the figure JSON
small.

## Chrome is per theme, palette colors are not

Background, text, line and grid colors differ between light and dark
themes. The color families, the categorical palette and the node card
colors are the same in every theme.

## Graph plots use a fixed pixel size per data unit

Node cards are sized in layout data units but their fonts in points. The
figure size of a graph plot is therefore derived from
`Theme.graph_px_per_unit`, so text fits its card at any graph size.
