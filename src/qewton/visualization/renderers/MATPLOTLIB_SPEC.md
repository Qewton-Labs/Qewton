# Matplotlib renderer: specification (temporary)

Adds a Matplotlib renderer, a renderer hierarchy based on the available
packages, and a notebook widget application. Once the feature is done, this
file moves into `REQUIREMENTS.md` and the renderer docs.

## Scope

**In scope:**
- Every plot whose primitives Matplotlib draws natively.
- Static outputs: image, HTML and GIF.
- At most one embedded control per output.
- Connected output through notebook widgets.

**Out of scope:**
- Parallel coordinates.
- Graph visualization.
- Hover and legend interaction.
- Linked 3D cameras.
- Changes to `pyproject.toml`.

## R: Renderer hierarchy

- **R1** The default renderer is the first renderer, in order of priority, whose package can be imported. The order is Plotly, then Matplotlib.
- **R2** Matplotlib is the guaranteed fallback.
- **R3** The library imports without Plotly, Dash or widget packages.
- **R4** A new renderer extends the hierarchy at a single place.
- **R5** A renderer or application whose package is missing fails with an `ImportError` that names the package.
- **R6** Renderers are passed as instances.

## M: Matplotlib renderer

- **M1** Implements the renderer and artist interface unchanged. Plot, Figure and control code do not depend on the renderer.
- **M2** Draws these primitives in 2D and 3D, where Matplotlib supports the dimension:
  - heatmap
  - surface
  - line
  - marker
  - bar
  - image
  - arrow
- **M3** Line and surface colors are continuous, with no binning.
- **M4** A primitive without an artist raises an error that names the Result type, the dim and the renderer.
- **M5** Supports faceted grids, overlays, and shared 2D position axes.
- **M6** Supports the axis features:
  - axis titles
  - categorical and log ticks
  - scaled (equal-aspect) channels
  - colorbars per color scale, including log scales
  - legends
- **M7** Artists are updated in place when the state changes; they are not recreated.
- **M8** Does not change global Matplotlib state and opens no window outside `show()`.

## E: Embedding

- **E1** The renderer declares which kinds of controls it embeds and how many per output. The Figure respects this declaration without type checks.
- **E2** Matplotlib embeds one TimeAxis or SliderAxis per output, as frames. A TimeAxis comes with playback.
- **E3** Controls that are not embedded are drawn in their current state and raise one `UserWarning` that names them and points to connected outputs.

## A: Applications

- **A1** An application declares the renderers it supports and rejects any other renderer.
- **A2** `WidgetApplication` shows a Matplotlib figure in a notebook (ipywidgets, ipympl):
  - a slider per SliderAxis
  - playback per TimeAxis
  - a choice per Selector
- **A3** On a change of state, the application updates the drawn figure without recreating it.
- **A4** Logic that applications share lives in the application base class.

## T: Theme

- **T1** Figures follow the theme:
  - colors
  - palette
  - fonts and font sizes
  - grid, axis box
  - line width, marker size, opacity
  - light and dark variants
- **T2** Qewton colormaps look the same as in Plotly. Renderer-native colormap names are accepted.
- **T3** Text has a LaTeX look without a LaTeX installation (Computer/Latin Modern fonts, math in `$...$`, also mixed with plain text). Real LaTeX rendering is available as an option.
- **T4** Theme settings apply per figure only.

## O: Outputs

- **O1** `save_image` writes PNG, SVG and PDF without external tools.
- **O2** `save_html` writes a self-contained page, animated if a control is embedded.
- **O3** `save_gif` writes the frames of the embedded control. Without one, it writes a single frame and warns.
- **O4** `show()` displays the figure in a notebook or in a window.

## Q: Tests

- **Q1** Matplotlib is a renderer in every test matrix. Omitted features appear in the expected-result tables, with a reason.
- **Q2** `WidgetApplication` is a connected output in the matrices.
- **Q3** Tests run headless.
- **Q4** A test checks the fallback to Matplotlib when Plotly is unavailable.
- **Q5** CI installs the widget packages; the existing single job is kept.
