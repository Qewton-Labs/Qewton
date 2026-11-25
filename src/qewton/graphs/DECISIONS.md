# Graphs: Design decisions

Grouped by the components of [REQUIREMENTS.md](REQUIREMENTS.md); each
heading names the requirements it serves.

## Places

### Configurations and values live in the graph, per place (PL1, PL3)

A node can take part in several graphs and at several places of one graph,
with different connections and sizes. The graph therefore keeps the
value of every port per place and a `Matching` (see `config`) with the
configuration of every port per place; ports and nodes hold neither. A node occurs twice in one graph through `node.copy()`,
which shares its weights.

### Removing matches again from the declarations (SU1)

Merging cannot be undone. Removing or replacing a node or an edge matches
the graph again from the unchanged static configurations.

A connection that fails does the same: a merge may have bound sizes
before it found the contradiction, so every way of adding an edge
restores the nodes and edges from before and matches again. Otherwise the
failed connection would still constrain the graph without an edge, and a
later valid connection would fail naming a node that is not connected.
Like removing, this matches without the refinements of a setup; the next
`setup` matches anew.

### One incoming edge per input port

An input port accepts at most one edge, an output port any number. An
unconnected input port uses its default; a port without default is
required and checked by `Graph.validate()`.

### Edges from outside do not count towards the in-degree

Edges created by `connect_from_outside_of_graph` carry data that already
exists when the graph starts (e.g. inside a `GraphNode`). They are ignored
when sorting, so the receiving node can run first.

## GraphNodes

### Inner graphs run in place (GN1)

The steps of a GraphNode's inner graph are inserted at its place, in the
inner graph's order. Running a graph is one flat list of steps, so nesting
costs nothing per run.

### A model with open sizes stays unbuilt (GN2)

Its inner graph depends on sizes that only the connected data provides.
Inferring them from the tensors of a direct call would bypass matching
and could disagree with other places, so a direct call before it is built
raises.

## Setup

### `setup` belongs to the library, nodes implement `build` (SU2)

`Node.setup` checks the state, requires known sizes and calls `build`,
which creates the algorithm (e.g. the network) separately from
`__init__`. A tuner makes a node unbuilt (`reset`) and sets it up again
without replacing it in its graph. Only `GraphAwareNode` and
`DataProcessingNode` receive the graph in `build(graph)`.

### `Graph.setup` matches anew (SU1)

Setting up nodes may change the graph: a `DataProcessingNode` runs it, and
nodes rebuild inner graphs. `Graph.setup` therefore matches and plans after
setting up all nodes. With open sizes it also matches before, so that they
are inferred before nodes are built. The inner graph of a GraphNode is set
up without a plan (`setup(plan=False)`): a graph containing the GraphNode
matches it in place, so a plan of its own is made only when it runs on its
own.

## Visualization

### Reference data in `Graph.visualize` is matched by Variable
Reference data must be described by a Variable equal to the one of the
visualized port (name, dim and children, see `config/DECISIONS.md`). The
model is evaluated at the reference's own points, so no interpolation is
needed for the comparison.

## Devices

### Nodes record their device
`Node.to(device)` records the device, so a node and a graph can report where
they live (`device`, None for a node without own data that follows its
inputs). A `ParameterNode` reports the device of its tensor once it is
initialized; a `GraphNode` and a `Graph` aggregate the devices of their
nodes. `Graph.visualize` then runs on the graph's device without moving
anything; with an explicit `device` it moves the nodes and moves them back
afterwards. A recorded device also lets `setup()` create parameters on the
right device when `to()` was called before it.

Status: planned. Today only a `DataNode` records its device, and
`Graph.visualize` without `device` creates mesh points on each sampler's
device.
