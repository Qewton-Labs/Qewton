# Config: Design decisions

Grouped by the components of [REQUIREMENTS.md](REQUIREMENTS.md); each
heading names the requirements it serves.

## Merging

### Declarations never change; matching has its own state (M1, M4)

Declared configurations are never written to. Everything matching finds
(groups of merged objects, bound ellipses, sizes, relations, origins) lives
in a `Matching`, one per graph, in tables over the declared objects. A node,
graph or trainer is then copied with a plain `deepcopy` (a tuner copies the
trainer per trial), and a new matching is built from the declarations
whenever the graph changes.

### Configurations of a matching are their own type (M1, M4)

`DynamicDataConfiguration` is a `DataConfiguration` that belongs to a
matching; merging and the shape operations exist only there. A declaration
thus has no operation that could change it, and everything that only reads
configurations takes both.

### Only open objects get a stand-in per place (M1, M4, H1, H2)

At a place, an object that can not become more concrete (a fixed size, a
dim bound to a HyperParameter, axes with a Variable or a discrete geometry
and only such dims) is used as it is. Every other object (an unknown dim,
an ellipsis, axes of unknown kind, features without a Variable, a
placeholder or continuous geometry, and what contains them) gets a
stand-in of its own, since it may become something else at every place,
e.g. features `u` at one place and `v` at another. What is one object in
the declarations stays one object at a place, and nothing is copied. A dim
bound to a HyperParameter is the same at every place, so all places agree
on its size.

### Merging instead of copying sizes (M1, G1)

Connecting merges axes and dims into groups instead of copying sizes
between separate objects. Equal axes are then one object where
configurations are read (`resolve()`), so consumers recognize them by
identity (one slider for the batch axes of a prediction and its reference)
and a size known at one place needs no further propagation. Of a group of
axes the most concrete one stands for it (a known kind, then a Variable or
geometry, a discretization before what it discretizes); of a group of dims
an operation dim, as its relation may still solve for its operands.

### Ambiguous alignments wait (M3, G4)

Of the consistent alignments of two sequences with ellipses, those that
merge the most are preferred; one remaining alignment is applied, several
wait until an ellipsis is bound. Guessing would make the result depend on
the order of `connect` calls.

## Relations of nodes

### `relate` is a pure function of shape operations (N1, G5)

A node that changes dims by their index states the change once, as a
function of its input configurations, with the operations a tensor would
use (`flatten`, `squeeze`, `window`, …). The operations reuse the input's
objects and express sizes as operation dims, so the way back needs no
recording: what the output learns, the input knows. Overriding a
propagation method per node would make every node author handle merging.

### Structure flows forward while connecting (N1, G4)

While connecting, a shape operation that would have to refine an unknown
input raises `StructureUnknown` and `relate` is run again once its inputs
change. Only `Graph.setup`, which matches from scratch, lets the matching
refine unknown inputs from known outputs (`Matching.refining`). Refinement
thus never meets structure arriving later.

### Broadcasting does not merge (N3)

A dim of size 1 broadcast against N is compatible with it but not the same
dim; merging them would give the size-1 side size N.

## HyperParameters in configurations

### Only plain open HyperParameters are inferred (H2)

A `HyperParameterOperation` (e.g. `hp * 2`) is read, never solved
backwards. A number in such an operation is a constant of the operation,
not a HyperParameter, so it is neither tuned nor inferred.

### `reset()` keeps an inferred value (H2)

The same node connected to other data is an error, not a silent change of
its weights' shape.

### Variables are values (M4)

Two Variables are equal if their names, dims and children are; copies
(e.g. of a trainer per trial) are equal to the original, and reference
data needs no particular instance. Composing Variables (`u * p`) leaves the
parts unchanged.

## Sizes of data sources

### Live sizes instead of matching again (D2, G2)

A dim bound to a HyperParameter reads its value each time, so changing a
batch size or switching to mesh mode needs no matching. The size found
while matching is used while no bound HyperParameter has a value.

### Run-time sizes are not checked (D3, G2)

The configured batch size describes the data; a smaller last batch runs as
it is. Checking it would cost time on every run.

## dtypes

### Unions and a numeric tower (T1)

Two dtypes match if they have a common member; the more special one wins
(`Any` → `TensorType` → backend tensor, `bool` → `int` → `float` →
`complex`). A union such as `TensorType | float` states that a backend
converts Python scalars itself.

## Saving

### dtypes are saved by structure (V1)

The loader only creates `qewton` classes. dtypes are therefore saved as
markers, names of builtins and backends, and unions, literals and
generics of these, and rebuilt from them.

### Older files are upgraded after loading (V2)

A file records its format version. After loading a file of an older
version, every object brings itself up to date (`Serializable.upgrade`):
nodes declare their ports anew from their annotations and add attributes
their class has since gained. The classes stay the only description of
their structure; there is no conversion of files.

### Matched configurations are not saved (V1)

They depend on the graph and are rebuilt from the static configurations
when the graph is matched after loading.
