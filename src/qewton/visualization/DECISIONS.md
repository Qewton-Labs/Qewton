# Visualization: design decisions

Grouped by the components of [REQUIREMENTS.md](REQUIREMENTS.md); each
heading names the requirements it serves. Decisions specific to a
subpackage are in its own `DECISIONS.md`.

## Plots and bindings

### Every dimension not drawn needs a control (P3, G6)

A plot draws the dimensions its positions cover and nothing else. Any
other dimension of the data must be reduced by a control, else
construction raises; nothing is sliced or averaged implicitly, so a plot
never shows a part of the data the user did not choose.

### Domain or values, decided by the reference (P3)

A binding's reference tells where positions come from: Variables and Axes
of the geometry or batch axes lie in the domain, features in the values.
A Variable held both by a geometry and the features (e.g. the grid a data
loader passes on) is read from the features, since the data holds its
values.

### Products are split by naming a factor (P4)

A product geometry knows its factors in the order of its tensor axes
(first factor slowest), so splitting it is a reshape:
`GeometryAxes.split` gives one axes per named factor and keeps adjacent
others as their product; `DataConfiguration.replace` describes the
reshaped data. A plot splits when a binding or control names a factor;
there is no separate restructuring step.

## Controls and selectors

### One control concept, several presentations (C1)

Slider, animation, facets, overlaid copies and a fixed state all reduce
one dimension to its states; they differ only in how the states are
shown. Switching the presentation is switching the class.

### Linking by instance (C2, SC2, F5)

One instance of a control, Selector, scale or position axis passed to
several plots or overlays links them: one widget, one range, one colorbar,
one zoom. Nothing is matched by name.

### Batch axes are linked only within one configuration (C2, GI2)

Data configurations of a graph are matched once at setup, and their axes
are copied, so two ports' batch axes cannot be recognized as the same
afterwards. Controls on batch axes are therefore shared only between
plots of one configuration; controls on Variables (geometries) are shared
everywhere. This changes with the rework of configuration matching.

## Scales

### Trained over every state (SC1)

Ranges are trained over all states of the controls before drawing, so
they do not jump while stepping. Position axes bound to a Selector of
table columns are the exception: their range follows the chosen column.

### Drawn to scale by reference (SC6)

Channels bound to components of one reference (e.g. the coordinates of a
geometry) are drawn to one scale, so geometries are not distorted; a
value over a geometry (e.g. a height) scales on its own.

## auto_plot and auto_layout

### Choice from declarations, explained when nothing fits (AU1, AU3)

`auto_plot` derives the kind of value (scalar, vector, coordinates,
values without domain) and the drawn domain, then takes the first entry
of a table of plot types and bindings whose type declares that it draws
the domain and accepts the given arguments. It knows no geometry classes;
a new plot type is a table entry. Where nothing fits, it raises with the
reason for every type tried instead of guessing.

### Default controls bind to Variables (AU2, GI2)

A default control on a geometry is bound to its Variable rather than to
the Axes object, so it fits every configuration of that Variable, e.g. the
reference and the prediction of `Graph.visualize`.

### Overlay by point encoding (AU4, P7)

Two plots on one domain hide each other when they show values at the same
points the same way (two color fields, two arrow fields). `auto_layout`
overlays plots only if no encoding occurs twice; otherwise it places them
side by side with shared position axes, so zoom and camera stay linked.

## Figure and layout

### Connected or embedded controls (F2, F3)

Each control is either connected to a widget of an application, which
sets its state and redraws, or embedded: the figure draws every state,
the renderer takes a snapshot of each and writes what differs. `show()`
and `save_html()` embed all controls; Dash connects sliders and selectors
and embeds the rest.

### What can be embedded together (F3)

An embedded control axis rewrites all data of its plot at each step.
Hence a plot embeds one control axis; next to it, one Selector drawn as
every candidate of which the chosen one is shown. Without a control axis,
several Selectors of different channels are embedded by their data, since
their steps change different properties. Anything else keeps its current
state and warns.

### A layout is composed once (F1, R1)

A figure takes its layout at construction and a renderer draws one
figure, so layouts are not edited afterwards; a changed arrangement is a
new figure.

## Renderers

### Theme and artists belong to the renderer (R1, T1)

What is drawn (plots, results, layout) is separate from how it looks: the
renderer owns the theme, the framework's figure and the artists. Results
carry an index into the categorical palette rather than a color.

## Graph integration

### Mesh-mode sizes come from the sampled geometry (GI1)

After a run in mesh mode, the axes of a sampler's geometry are sized to
the points the sampler provides now; they are found by the instance of
its `SampledGeometry`, not by configuration matching.

### Panels are named by legend or title (GI2)

Reference, prediction and error are named by their legend entries in one
overlay and by their titles side by side, as `auto_layout` places them.
