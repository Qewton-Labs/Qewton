# PLAN (temporary): Matching data configurations in graphs

The architecture that meets [SPEC.md](SPEC.md). Status: **planned** (2026-11-17); every
part is decided, with its date. Decisions move to DECISIONS of `config` and `graphs`
once done; this file is deleted afterwards.

Reference implementation of A1–A2: the spike in the worktree
`../pioneer-backend-spike` (branch `spike-node-uses`), with
`tests/test_graphs/test_matrix_node_reuse.py`.

## A1 Node, Port, Graph and places (decided 2026-11-16)

- A **Node** computes, in `forward` only; there is no `Node.run`.
- A **Port** declares: name, static configuration, default. It holds no
  value.
- A **Graph** keeps the matched configuration and the value of every port
  it contains, per **place** = (path of GraphNodes, port). Queries:
  `graph.config_of(port, within=...)` (also via
  `port.get_data_configuration(graph, within=...)`) and
  `graph.value_of(port, within=...)`. `within` is needed only where a node
  occurs at several places. Both stores are internal and not saved;
  configurations are rebuilt after loading.
- A node occurs at several places of one graph through `node.copy()`
  (`CopiedNode`, sharing the weights). While tracking a function
  (`FromFunctionNode`, `TrackedNode`), a node called again is copied this
  way automatically. Shared weights are collected once.
- Losses are read through the graph: `constraint.get_loss(graph)`.

Meets R4, R5 (structure), S1, S8.

## A2 GraphNodes in a graph (decided 2026-11-16)

- In a graph, the steps of a GraphNode's inner graph are inserted at its
  place, in the inner graph's own order (no re-sorting). Inputs read the
  place of the output that computes them; the ports of the GraphNode only
  lead there. Running a graph is one flat list of steps (R19, P6).
- The GraphNode and its inner graph stay the objects the user built:
  devices, modes and `graph.diagram` work on them.
- `GraphNode.forward` evaluates the inner graph outside of graphs;
  subclasses override `forward` only to declare annotations, with the body
  `return super().forward(...)`.

## A3 Matching as type inference (decided 2026-11-16)

- A node's annotations declare configurations with unknowns (size `None`,
  ellipsis, open Variable). At every place of a graph the declaration is
  instantiated once; objects shared within the node stay shared (R3).
- Connecting **merges** corresponding axes and dims into one object and
  binds an ellipsis to a sequence of axes, possibly empty. A contradiction
  is an error at connecting.
- Technique: union-find while matching; a closing pass then replaces every
  reference by its representative, so `is` holds whenever configurations
  are observed (after `connect` and `setup`).
- Geometries and Variables are never changed; only the per-place axes
  objects referring to them are merged (R9).
- Work list instead of recursion: what depends on a newly bound unknown is
  matched again, nothing else (R17). An ambiguous ellipsis equation is
  deferred until more is known (R8, P5).

## A4 Relations of axis-changing nodes (decided 2026-11-17)

- Annotations relate ports by shared objects and arithmetic expressions
  of dims (`n + m`, `d * d`).
- Axis-changing nodes add a pure function `relate(...)` on symbolic
  configurations, built from *shape* primitives that reuse the input's
  objects: `drop` (`squeeze` = drop of size 1), `set_size`, `unsqueeze`,
  `permute`, `reshape` / `flatten` / `unflatten`, `concat` / `split`,
  `broadcast`, `window(kernel, stride, padding, dilation)`. Reductions are
  `drop` or `set_size(…, 1)`.
- The way back follows from the recorded primitives where it is unique;
  otherwise the size stays unknown and is checked once both sides are known
  (window with stride > 1, broadcast, slicing). Free Python code in
  `relate` runs forward only, with an optional hand-written inverse.
- Element-wise operations with several inputs use `broadcast` (R11).

Existing nodes and their relation:

| Group | Nodes | Relation | Way back |
|---|---|---|---|
| element-wise, one input | activations, `Sin`, `Exp`, `Abs`, `Identity`, `GradientTracking`, `ZerosLike`, … | shared ellipsis | full |
| element-wise, several inputs | `Add`, `Subtract`, `Multiply`, `Divide`, `Power`, `Maximum`, … | `broadcast` | partial |
| reductions | `Sum`, `Mean`, `Std`, `Norm`, `Inner` | `drop` / `set_size(1)` | partial (reduced axis open); full with keepdims |
| reorder | `Squeeze`, `Unsqueeze`, `Transpose` | `squeeze`, `unsqueeze`, `permute` | full |
| reshape | `Reshape`, `ReshapeAtDim`, `Flatten`, `Unflatten` | `reshape`, `flatten`, `unflatten` | full with known factors (today: no relation at all) |
| join / split | `ConcatNode`, `ConcatVariables`, `SplitVariables` | `concat`, `split` | partial / full |
| cut / repeat | `Narrow`, `Slice`, `Repeat` | `set_size`, indexing, repeat | forward only / forward only / full |
| linear algebra | `MatMul`, `Dot`, `Linear`, `FunctionalLinear` | shared dims | full |
| derivatives | `Gradient`, `Laplacian`, `Jacobian`, `Hessian`, … | shared ellipsis and dims | full |
| convolution | `Conv`, `MaxPool*`, `AvgPool*`, `Interpolate` | `window` | full with stride 1, else forward only |
| other | `SVD`, `MeshInterpolationNode`, `GetShapeNode` | min of dims / data-dependent | partial / none |
| spectral | `FunctionalSpectralConv` | shared batch and geometry axes, channel dims bound to hyperparameters (A5); FFTs stay inside `forward` | full |
| embeddings | `GridEmbeddingND` | channels `n + dim` | full |
| GraphNodes | FCN, DeepONet, CNN, Encoding, PCA, FNO, `FNOBlock`, `SpectralConv`, `ChannelMLP`, `SoftGating`, … | their inner graph (A2) | as their inside |

## A5 Hyperparameters in configurations (decided 2026-11-17)

- A HyperParameter fixes how a node is built and is not learned. Its value
  comes from the user (`FIXED`), from a tuner (`OPTIMIZE`) or from the
  connected data (`INFERRED`, new).
- A dim may be bound to a HyperParameter, e.g.
  `FeatureAxes(shape=(self.in_neurons,))`. Such a dim belongs to the node:
  it is not instantiated per place but is the same object at every place.
  Matching fills an open HyperParameter like any other unknown; places that
  disagree are an error at connecting. Arithmetic of HyperParameters
  (`HyperParameterOperation`) and of dims express the same relations.
- Sizes that depend on the place (batch, number of points) are plain dims,
  instantiated per place; they never flow into the state of a node.
- Replaces the overrides that derive node state from configurations
  (Encoding `input_shape`, DeepONetCNN/FCN `branch_input`); `Slice` and
  `SplitVariables` positions follow from `relate` (A4).
- A tuner setting a HyperParameter changes the dims bound to it; only what
  depends on them is matched again.

## A6 Setup (decided 2026-11-17)

- Order: build the node (hyperparameters may be open) → connect (match per
  place, infer open hyperparameters, A5) → `setup()` → run. A tuner sets
  hyperparameters and continues at `setup()`.
- `Node.setup()` belongs to the library (template method). In a fixed
  order it checks the state (no rebuild of an initialized node), makes
  sure the hyperparameters are concrete, calls `self.build()`, reports
  changed static configurations for matching (R12), discards the plan of
  a GraphNode's inner graph and sets the state. Node authors override only
  `build()` (`build(graph)` for `GraphAwareNode` and `DataProcessingNode`).
  A subclass overriding `setup` is rejected with a message pointing to
  `build` (`__init_subclass__`).
- `Graph.setup()` sets up all nodes, then plans (matches and lists the
  steps), always anew: a `DataProcessingNode` runs the graph during its
  setup, and nodes may rebuild inner graphs.
- Users set up a graph again after setting up single nodes by hand.
- About 25 nodes rename `setup` to `build`; 14 of them lose their own
  state check.

## A7 Changing sizes (decided 2026-11-17)

- Sizes that depend on the place come from the HyperParameters of data
  sources (`batch_size`, `n_points`, discretization). The dims behind them
  are bound to these HyperParameters (A5), so a change holds everywhere at
  once, without matching.
- A batch dim has the configured size as its value. The actual size at run
  time may differ (last batch); it is neither checked nor written back.
- Plotting in mesh mode switches the sampler's discretization, which
  changes the bound dims; `Graph._with_current_sizes` is no longer needed.

## A8 dtypes, errors, data sources, saving (decided 2026-11-17)

- dtypes form an order from general to special: `Any` → `TensorType` →
  backend tensor → concrete dtype (`float32`, `complex64`). Matching keeps
  the more special one; incompatible ones are an error. A dtype belongs to
  a port (complex weights next to real inputs are fine); nothing is
  overwritten in place (R13).
- Every representative of the union-find remembers the port that brought
  in its value; a contradiction names both origins, e.g. "batch 7 from
  sampler `inner` and 5 from sampler `boundary` meet at `FCN.x`" (R14).
- Data loaders and samplers declare one batch axes for all their outputs,
  bound to `batch_size` / `n_points` (A5, A7) (R15).
- Matched configurations are not saved but rebuilt after loading (A1);
  static configurations are saved with their dtype (R16).

## Findings

- Spike measurements (`main` → spike): run 886 → 802 µs (PINN pipeline),
  156/174/187 → 111/108/116 µs (FCN in 0/2/4 wrapping GraphNodes); build
  2.5 → 11 ms and 13–15 → 38–62 ms, because configurations are copied and
  matched per place (to be removed by A3).
- Bugs found on `main`: `replace_node` breaks the replaced node (xfail in
  `test_graphs`); `TrackedNode` cannot be set up (no test existed; fixed in
  the spike); dtypes of saved configurations are lost.
- FNO (merged 2026-11-17 from `main`): its five GraphNodes use the
  `forward` boilerplate replaced by A2; `FNO.setup` rebuilds without a state
  check, so with several graphs every `Graph.setup()` would re-initialize
  the weights (fixed by A6). No tests exist for FNO and its building blocks.
- `examples/PINN/stokes.ipynb` exceeds 30 minutes even with 3 iterations,
  on `main` as in the spike.
