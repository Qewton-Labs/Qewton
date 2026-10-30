# Matplotlib renderer: tasks (temporary)

These are the tasks for [MATPLOTLIB_SPEC.md](MATPLOTLIB_SPEC.md) and
[MATPLOTLIB_PLAN.md](MATPLOTLIB_PLAN.md).

Every step ends with the full suite green and a review before the next step
starts. IDs in brackets refer to the spec.

## Step 1: Hierarchy and declarations (Plotly only, no visible change)

- [ ] **1.1 Lazy application imports** [R3, R5]
  - Add `require(*packages)` to `applications/base.py`.
  - Import `dash` inside `DashApplication.create`.
  - `applications/__init__.py` and `visualization/__init__.py` import without Dash.
- [ ] **1.2 `RenderApplication` refactor** [A1, A4]
  - Add `connected`, `renderer_classes()` and `check_renderer()` to the base class.
  - Move `interactive_controls`, `embedded_controls` and `update` from Dash into the base class.
  - Dash declares `connected = (SliderAxis, Selector)`.
  - A wrong renderer raises a `TypeError`; adjust the tests that expect the `ValueError`.
- [ ] **1.3 Embedding declaration** [E1, E3]
  - Add `Renderer.embeddable` and `max_embedded`, and declare them in Plotly.
  - `Figure._embeddable` skips controls the renderer does not embed, or once the limit is reached.
  - Make the warning text generic, and adjust the tests that match it.
- [ ] **1.4 Renderer hierarchy** [R1, R4]
  - Add `RENDERER_DICT` with Plotly in a `try` block.
  - Export `PlotlyRenderer` only when it is available.

## Step 2: Static Matplotlib renderer

- [ ] **2.1 `style.py`** [T1, T2, T4]
  - `rc_params(theme, usetex)`
  - `colormap(resolved)`, cached
  - `norm(scale)`
  - Tests: the rcParams of the light and dark theme, Qewton and native colormaps, LogNorm.
- [ ] **2.2 `text.py`** [T3]
  - `as_mathtext`: plain-text fallback with a `UserWarning`, cached.
  - Tests: valid math, mixed text, an unknown command.
- [ ] **2.3 Renderer core** [M1, M5, M6, M8, O1, O4, R2]
  - `MatplotlibRenderer`: `__init__` (`usetex` check), `style()`, `create_backend_figure` (grid, 3D, shared axes, cell labels), `draw`, `draw_axes`, legend, `draw_colorbar`.
  - Outputs: `save_image`, `html` (static SVG), `save_html`, `show`.
  - Register Matplotlib last in `RENDERER_DICT`.
  - Set `MPLBACKEND=Agg` in `tests/conftest.py`.
- [ ] **2.4 2D artists** [M2, M3, M7]
  - Base class `MatplotlibArtist` and `axes_of`.
  - Artists: Heatmap, FlatMesh, Line2D, Marker2D, Bar, Image, Arrow2D.
- [ ] **2.5 3D artists** [M2]
  - Surface, Line3D, Marker3D, Arrow3D.
- [ ] **2.6 Tests** [Q1, Q3, Q4, M4]
  - Add `"matplotlib"` to `RENDERERS` of every matrix, for the `draw` output.
  - Add expected-result entries for parallel coordinates and graphs.
  - New `test_matplotlib.py` with the rules: rc not leaked, mathtext labels, LogNorm colorbar, shared axes, PNG, SVG and PDF.
  - New `test_renderers.py`: the fallback to Matplotlib in a subprocess with Plotly blocked.

## Step 3: Embedding

- [ ] **3.1 Snapshots and animation** [E2]
  - Matplotlib declares `embeddable = (TimeAxis, SliderAxis)` and `max_embedded = 1`.
  - Implement `snapshot`, `animate`, `apply_slider` and `clear_embeddings` (`self.animation`).
- [ ] **3.2 Animated outputs** [O2, O3]
  - `html()` via `to_jshtml()`.
  - `save_gif` with `PillowWriter`; the frame weights become repeated frames.
- [ ] **3.3 Tests** [Q1, E3]
  - The self-contained column of the matrices for Matplotlib.
  - Expected warnings for a second control or a Selector.
  - The GIF frame count.

## Step 4: WidgetApplication

- [ ] **4.1 `widget_app.py`** [A2, A3]
  - `WidgetApplication`: an ipympl canvas of the existing figure.
  - One widget per control: an IntSlider per SliderAxis, a Play widget with a slider per TimeAxis, a Dropdown per Selector.
  - A change calls `update` and then `canvas.draw_idle()`.
  - Prerequisite: `ipympl` in the `pioneer` env (not installed yet).
- [ ] **4.2 Tests** [Q2]
  - Add `APPLICATIONS = {renderer: application}` to the matrices, for the connected column.
  - Use `importorskip` for ipywidgets and ipympl.
- [ ] **4.3 CI** [Q5]
  - Add `ipywidgets ipympl` to the install line in `.github/workflows/main.yml`.

## Step 5: Documentation and cleanup

- [ ] **5.1 Docs** (in English, written as API docs)
  - Move the spec into `REQUIREMENTS.md`, with a status per requirement.
  - Update the `renderers/` README and DECISIONS: the hierarchy, Matplotlib and what it leaves out.
  - Update the visualization README: choosing a renderer, the applications.
- [ ] **5.2 Cleanup**
  - Delete `MATPLOTLIB_SPEC.md`, `MATPLOTLIB_PLAN.md` and `MATPLOTLIB_TASKS.md`.
  - Run the full suite.
