# Visualization: Requirements

What the visualization package must achieve (Part 1, independent of any
solution) and its architecture with the requirements on each component
(Part 2). [README.md](README.md) and [DECISIONS.md](DECISIONS.md) follow
the same components; decisions refer to requirements by ID.

# Part 1: General requirements

## Context

Qewton produces data described by a `DataConfiguration` (its axes: batch,
geometry and features), geometries and meshes, computation graphs, and
training and tuning logs. Plotting frameworks with different strengths
exist (e.g. Plotly, Matplotlib, VTK, trame, ParaView) and are to be used
rather than replaced. A Qewton frontend will be built on top.

## Scenarios

| ID | Scenario |
|---|---|
| S1 | Explore results inline in a notebook. |
| S2 | Export figures and animations for reports and papers (HTML, PNG, SVG, GIF). |
| S3 | Compare a model with reference data (prediction, reference, error). |
| S4 | Inspect data interactively in a live application. |
| S5 | Use an application that builds its views itself (e.g. trame, ParaView) from Qewton data. |
| S6 | Serve visualizations in a Qewton frontend, which runs actions inside Qewton. |
| S7 | Monitor training and tuning runs, by plotting statistics and intermediate results. |
| S8 | Visualize geometries and meshes, e.g. for finite element models. |
| S9 | Visualize computation graphs, e.g. for debugging or teaching. |
| S10 | Get a quick first visualization without configuration, and fully customize it for publication-quality figures. |
| S11 | Usage by scientists and engineers, leaning towards mathematically clear concepts. |

## Requirements

| ID | Requirement | Scenarios |
|---|---|---|
| G1 | Everything Qewton produces can be visualized: data of any structure (grids, meshes, point clouds, curves, samples, products of domains such as space × time, batches), geometries, graphs and run logs. | S1–S9 |
| G2 | Dimensions that cannot be drawn at once can be explored: stepped through, animated, shown side by side, fixed, or switched between quantities. | S1, S4, S6, S7 |
| G3 | Every aspect of a visualization can be customized. | S2, S10 |
| G4 | Results can be compared with references in a consistent way: same views, same controls, same value ranges, plus the difference. | S3, S7 |
| G5 | The same visualization can be shown inline, exported, served live, used by external applications and by the frontend. | S1, S2, S4, S5, S6 |
| G6 | Nothing requested is omitted silently: what an output cannot show is shown in a defined fallback, with a notice. | all |
| G7 | Several plotting frameworks can be used and combined; adding one does not require changes elsewhere. | S5, S6, S8 |
| G8 | Visualizing works wherever the computation runs (e.g. CPU or GPU) without extra handling by the user. | S1, S3, S7 |
| G9 | Large data stays usable: interactive views stay responsive, exported files stay of reasonable size. | S2, S4, S6, S7 |
| G10 | Every visualization fits to a pre-defined visual appearance of Qewton, e.g. colors, fonts, line widths, and figure size. | all |
| G11 | Features combine freely: every combination of data, view, control, layout, framework, appearance and output works, except implausible ones, which are rejected with a clear error. | all |

# Part 2: Architecture

A *plot* binds references into its input (Variables, Axes, or table
columns) to the channels of one kind of mark and evaluates to
renderer-neutral *results*. *Controls* reduce the dimensions a plot does
not draw; *scales* map values to positions, colors and arrow lengths. A
*figure* arranges plots in a layout of cells and draws them with a
*renderer*, which owns the theme and the drawing framework. An output
either *embeds* the states of the controls (self-contained) or *connects*
them to widgets of an application.

Status: done, partial (with the gap named), open.

## Plots and bindings (`plots/`)

| ID | Requirement | From | Status |
|---|---|---|---|
| P1 | A plot evaluates, in the current states of its controls, to one Result per drawn copy; Results carry no framework types. | G5, G7 | done |
| P2 | Plot types declare what they draw: channels, numbers of positions, the domains they draw (structure and topological dimension of the geometry). Nothing else repeats this knowledge. | G1, G11 | done |
| P3 | Positions come from the domain (coordinates of a geometry or the index of a batch) or from the values; every dimension of the data that is not drawn is reduced by a control, else construction fails. | G1, G6, G11 | done |
| P4 | Data on a product geometry is drawn by naming a factor in a binding or control; the plot splits the axes of the product. | G1, G2 | done |
| P5 | Data of any backend and device is drawn as given. | G8 | partial: a node's device is not yet its own property (strict xfail) |
| P6 | Families: data (`DataConfiguration`), tables (named columns), geometries, computation graphs; one Plot interface for all. | G1 | done |
| P7 | A plot declares how it shows values at the points of its domain (color, arrows, by position), so layouts can tell which plots read well together. | G4, G11 | done |

## Controls and selectors (`axes.py`)

| ID | Requirement | From | Status |
|---|---|---|---|
| C1 | A control axis reduces one dimension; kinds differ only in presentation: slider, animation, facets, overlaid copies, fixed state. | G2 | done |
| C2 | One control instance passed to several plots links them. | G2, G4 | done |
| C3 | A Selector chooses which reference fills a channel (Variables of the data or table columns). | G2 | done |
| C4 | States are labeled by coordinates or categories; animation frames last in proportion to their time step. | G2, G3 | done |
| C5 | A control steps through a whole one-dimensional geometry, a factor of a product, a batch axis or a table column. | G2 | partial: one axis of a multi-axis grid geometry is not supported |

## Scales (`scales.py`)

| ID | Requirement | From | Status |
|---|---|---|---|
| SC1 | Scales are trained over every state of the controls, so ranges stay fixed while stepping. | G2, G4 | done |
| SC2 | One scale instance shared by several plots gives one range and one colorbar. | G4 | done |
| SC3 | Fixed bounds, symmetric and logarithmic scales; log scales show values, not exponents. | G3 | done |
| SC4 | A Selector gets one color scale per candidate by default, or a shared or given one per candidate. | G2, G3 | done |
| SC5 | Arrows are scaled to lengths in data units by a VectorScale. | G1, G3 | done |
| SC6 | Components of one reference (e.g. the coordinates of a geometry) are drawn to one scale; other channels scale independently. | G1, G10 | done |

## auto_plot and auto_layout (`auto.py`, `layout.py`)

| ID | Requirement | From | Status |
|---|---|---|---|
| AU1 | `auto_plot` chooses the plot type from the declarations of the plot types, the kind of value and the drawn domain, without knowing geometry classes. | G1, G10, G11 | done |
| AU2 | Dimensions not drawn get a default control; products are split where needed. | G2, G10 | done |
| AU3 | Plot arguments take part in the choice; where nothing fits, the error names each plot type tried with its reason. | G6, G10 | done |
| AU4 | `auto_layout` overlays plots that read well together and places the others side by side with linked axes. | G4, G10 | done |

## Figure and layout (`figure.py`, `layout.py`)

| ID | Requirement | From | Status |
|---|---|---|---|
| F1 | Overlay, Row and Column compose a grid of cells; facets repeat a cell; irregular grids are padded. | G3 | done |
| F2 | Outputs: drawing, self-contained (shown inline, HTML, images, GIF) and connected (applications). | G5 | done |
| F3 | A self-contained output embeds the states of the controls; controls that cannot be embedded together on one plot keep their current state, with a warning. | G2, G6 | done |
| F4 | Embedding stays usable for large data. | G9 | partial: about 15 ms per embedded state |
| F5 | Cells sharing their position axes zoom and rotate together; a control does not reset the view. | G4 | done (Plotly: not in its own notebook output) |
| F6 | Titles of cells, axes, colorbars and legends are derived from the references. | G3, G10 | done |

## Renderers (`renderers/`)

| ID | Requirement | From | Status |
|---|---|---|---|
| R1 | A renderer draws one figure in one framework and one theme; artists draw Results, one per Result class and number of positions. | G7, G10 | done |
| R2 | A renderer embeds states by writing what differs between them. | G2, G9 | done |
| R3 | Further renderers need no changes elsewhere. | G7 | done (Plotly is the only renderer) |
| R4 | Names of Variables are written in math mode where the framework renders it. | G10, S11 | done |

## Applications (`applications/`)

| ID | Requirement | From | Status |
|---|---|---|---|
| A1 | A live application connects sliders and selectors to widgets and embeds the other controls. | G5, S4 | done |

## Themes (`themes/`)

| ID | Requirement | From | Status |
|---|---|---|---|
| T1 | A theme sets every visual setting; colormap roles resolve against it. | G10 | done |

## Graph integration (`qewton.graphs`, `qewton.data`, `nodes.py`)

| ID | Requirement | From | Status |
|---|---|---|---|
| GI1 | `Graph.visualize` evaluates a port on a deterministic discretization of its samplers and draws it. | G1, S1 | done |
| GI2 | With a reference (data, port or function), reference, prediction and error share views, controls and color range. | G4, S3 | partial: controls on batch axes are shared only within one configuration |
| GI3 | Data nodes and `PlotNode` draw their own data. | G1, S7 | done |
| GI4 | Computation graphs are drawn as node-link diagrams. | S9 | done |
