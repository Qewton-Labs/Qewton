# Design Decisions: `graphs`

## Unified data configurations live in the graph
Connecting two ports unifies their data configurations, but the result is
stored in `Graph.dynamic_data_configs`, not in the ports. The same node can
take part in several graphs with different connections, and removing a node
or edge can recompute all configurations from the unchanged static ones.

## One incoming edge per input port
An input port accepts at most one edge, an output port any number. An
unconnected input port uses its default; a port without default is
required and checked by `Graph.validate()`.

## Edges from outside do not count towards the in-degree
Edges created by `connect_from_outside_of_graph` carry data that already
exists when the graph starts (e.g. inside a `GraphNode`). They are ignored
when sorting, so the receiving node can run first.

## `Node.setup()` is separate from `__init__`
The underlying algorithm (e.g. a neural network) is created in `setup()`,
not in the constructor. Hyperparameter tuning recreates the algorithm
without replacing the node in its graph.

## Graph-aware setup is opt-in
Only `GraphAwareNode` (and `DataProcessingNode`) receive the graph in
`setup(graph)`, e.g. to read the unified configurations of their ports.
All other nodes keep the plain `setup()` signature.

## Reference data in `Graph.visualize` is matched by Variable identity
Plain reference data must be built from the exact same `Variable` instance
as the visualized port; a reference port only needs a structurally equal
Variable. Matching by name would silently accept unrelated quantities. The
model is evaluated at the reference's own points, so no interpolation is
needed for the comparison.
