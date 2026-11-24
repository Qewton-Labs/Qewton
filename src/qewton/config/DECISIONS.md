# Config: Design decisions

Grouped by the components of [REQUIREMENTS.md](REQUIREMENTS.md); each
heading names the requirements it serves.

## Merging

### Merging instead of copying sizes (M1, G1)

Connecting merges axes and dims with union-find instead of copying sizes
between separate objects. Equal axes are then one object, so consumers
recognize them by identity (one slider for the batch axes of a prediction
and its reference) and a size known at one place needs no further
propagation. `resolve()` replaces merged objects by their representatives
where configurations are read.

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
change. Only `Graph.setup`, which matches from scratch, refines unknown
inputs from known outputs. Refinement thus never meets structure arriving
later.

### Broadcasting does not merge (N3)

A dim of size 1 broadcast against N is compatible with it but not the same
dim; merging them would give the size-1 side size N.

## HyperParameters in configurations

### A bound dim is copied per place (H1, H2)

Merging writes into dims, so a dim shared between places would carry the
matching of one graph into another. Every place gets its own copy; all
copies refer to the same HyperParameter, which keeps them equal.

### Only plain open HyperParameters are inferred (H2)

A `HyperParameterOperation` (e.g. `hp * 2`) is read, never solved
backwards. A number in such an operation is a constant of the operation,
not a HyperParameter, so it is neither tuned nor inferred.

### `reset()` keeps an inferred value (H2)

The same node connected to other data is an error, not a silent change of
its weights' shape.

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
