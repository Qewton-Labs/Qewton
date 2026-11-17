# SPEC (temporary): Matching data configurations in graphs

What the rework of data configuration matching must achieve and how it is
accepted. The architecture that achieves it is in [PLAN.md](PLAN.md).
Status: **specified** (2026-11-17). Requirements move to
REQUIREMENTS of `config` and `graphs` once done; this file is deleted
afterwards.

## Goal

Connecting two ports matches their data configurations (shapes, axes,
variables, dtype) and makes the whole graph more concrete. Axes and
dimensions that are the same are the same *objects* everywhere in a
graph, so that

- consumers recognize equal axes by identity (e.g. one slider for the
  batch axes of a prediction and its reference),
- a size set at one place holds everywhere without further matching.

## Scenarios

| ID | Scenario |
|---|---|
| S1 | PINN: one model (`FCN`, a `GraphNode`) is used in several pipelines (PDE, boundary, initial condition) with different samplers and batch sizes; weights are shared. One sampler can feed several pipelines. |
| S2 | DeepONet: a `GraphNode` whose inner graph (branch, trunk, merger) is built in `setup()`; branch input size is inferred from the connected data. |
| S3 | Plot a port with reference data: prediction, reference and error share one control per batch axis. |
| S4 | Plot in mesh mode: samplers provide a different number of points than during training. |
| S5 | Tuning: hyperparameters become concrete late; `setup()` rebuilds the algorithm, possibly with other port configurations. |
| S6 | Inspect the configurations inside nested graphs, e.g. with `graph.diagram`. |
| S7 | Data pipelines: data loader → processing (normalization, PCA) → model, train/val/test with different batch sizes. |
| S8 | Weight sharing: one model evaluated at several places of one graph, e.g. called twice in a tracked function. |

## Principles

| ID | Principle |
|---|---|
| P1 | Connecting *merges* axes and dimensions instead of copying sizes between separate objects. Identity is the result of matching. |
| P2 | No checking at run time: all matching happens while building or setting up a graph; running a graph has no overhead from configurations. |
| P3 | Identity is the query API: equal axes/dims are the same object, compared with `is`. No separate "are these the same" lookup. |
| P4 | Few, clear base rules; user-friendliness comes from structures built on them, not from special cases. |
| P5 | The result of matching does not depend on the order of `connect` calls. |
| P6 | Deeply nested `GraphNode` graphs are the normal case. Running them adds no per-run overhead from ports or nesting. |
| P7 | Writing a new node is comfortable for external developers (open-source community, extensibility): the rules a node author must know are few and local. |

## Requirements

*Today*: ✅ fulfilled, ⚠️ partial, ❌ missing (state of `main`).
*Acceptance*: the criterion, and the tests that check it; tests in
*italics* are still to be written. Node and model registries of the test
matrices include every built-in model. FNO is accepted through
`examples/OperatorLearning/FNO/darcy_flow.py` (compares with neuralop; needs
`neuraloperator` and internet) until real tests exist.

| ID | Requirement | Today | Acceptance |
|---|---|---|---|
| R1 | Connecting two ports unifies their configurations; the result reaches every port of the graph it concerns, also inside GraphNodes. | ⚠️ by copying; does not pass through `Linear` or a nested GraphNode | A size given at one end of a chain of nodes, including nested GraphNodes, is known at the other end. `test_graph_config_propagation`, `test_matrix_node_reuse` (`outer_config`, `inner_*_config`) |
| R2 | Equal axes (`Axes`) and equal dimensions (`AxesDim`) are the same object across edges and across the boundary of GraphNodes. Same Axes = semantically the same axis; same dim = same size only. | ❌ sizes are copied between objects | The batch axes of a sampler, of the model input and of the model output are one object (`is`). *`test_matrix_identity`*: chains × node kinds |
| R3 | A node declares which axes it passes through by using one Axes object in several port annotations. | ✅ deepcopy memo | as R2, for every node of the registry |
| R4 | A node may take part in several graphs, and at several places of one graph; configurations and values are kept per place. Weights and other state of the node stay shared. | ⚠️ configs per graph; values on shared ports; not twice in one graph | `test_matrix_node_reuse` (uses × checks, `TestCopiesInOneGraph`, `TestTrackedFunctions`) |
| R5 | A GraphNode in several outer graphs is matched consistently with each of them; its inner configurations are inspectable per place (S6). | ❌ one inner graph, overwritten by every outer graph | `test_matrix_node_reuse` (`inner_input_config`, `inner_output_config`); *`graph.diagram` shows inner configurations of a place* |
| R6 | Nodes that change axes express this through one defined mechanism. | ⚠️ about 10 overrides of `update_data_configs` | No built-in node overrides `update_data_configs`. *`test_matrix_relate`*: node registry × known side (input / output) × expected other side |
| R7 | Sizes can depend arithmetically on others (sum, product, …) and are resolved in every direction where unique. | ⚠️ `OperationDim`, one-way | *`test_matrix_relate`* rows for concat, reshape, flatten, window with stride 1 |
| R8 | An ellipsis binds to a sequence of axes/dims, possibly empty; a shared ellipsis stands for the same sequence. An ambiguous match is deferred, not guessed. | ⚠️ heuristic `_match_remainder` | *`test_matrix_ellipsis`*: ellipsis cases (empty, one, several, on both sides) × all orders of `connect` give the same result (P5) |
| R9 | Matching never changes shared objects: geometries and Variables stay shared between graphs (no copies), static configurations stay untouched, no node state is derived per graph. | ❌ `FeatureAxes.unify_with`, `Geometry.unify_with` mutate; four nodes store graph-derived state | Static configurations, geometries and Variables are equal before and after building two graphs from the same nodes. *`test_matrix_no_mutation`* |
| R10 | Sizes that depend on the place change only through the HyperParameters of data sources (batch size of a data loader, number of points or discretization of a sampler); every dim bound to them changes at once, without matching (P2). A batch dim keeps its configured size as its value; at run time the actual size may differ (e.g. a smaller last batch), and nothing is checked or adapted then. | ❌ `_with_current_sizes` builds new axes for plotting | Changing the batch size of a data loader (or the points of a sampler) changes the configuration of every port behind it, including inside GraphNodes, without another `connect` or `setup`; a smaller last batch runs. Plotting in mesh mode uses the configurations as they are. *`test_matrix_sizes`*: sources × change × ports behind |
| R11 | Broadcasting (size 1 vs. N) makes dims compatible but does not merge them; element-wise operations with several inputs relate them by broadcasting. | ⚠️ `Add` & co. declare one shared ellipsis | `x` of batch N and `y` of batch 1 connect to `Add`; the output has batch N; `x` and the output share the batch axes, `y` does not. *`test_matrix_relate`* |
| R12 | Structural changes (remove, replace node, rebuilt inner graph in `setup()`) leave consistent configurations and a valid evaluation order. | ⚠️ not re-matched after `setup()`; `replace_node` breaks the replaced node | `test_graphs` (`replace_node` tests); `test_fcn` (HarmonicFCN setup); examples with `DataProcessingNode` (PCANets) |
| R13 | dtypes are matched including subtypes (`Any`, `TensorType`, backend dtypes). | ⚠️ equality only, mutates `dtype` | *`test_matrix_dtypes`*: dtype pairs × expected result or error |
| R14 | A mismatch error names the connection and the chain of ports that led to the conflict. | ⚠️ nested messages | *Error messages of `test_matrix_relate` / `test_matrix_ellipsis` contradiction rows name both ports.* |
| R15 | Data sources share axes between their outputs where the data shares them (e.g. one batch axes for x and y of a data loader). | ❌ one `BatchAxes` per output port | *Data loader and samplers: outputs share the batch axes (`is`).* |
| R16 | `duplicate`, saving and loading keep the structure; configurations are rebuilt from it after loading. | ⚠️ dtypes of saved configurations are lost | `test_save_graph` (all models load and run) |
| R17 | Building is efficient for large, deeply nested graphs: no repeated full passes. | ⚠️ recursive propagation | Building time grows at most linearly with the number of nodes; benchmark of nested FCNs (spike: build 3–5× slower than `main` must be undone). |
| R18 | A node declares the relations between its ports only in its annotations and, for axis-changing operations, in a pure relation function; the relations hold in every direction where unique. | ⚠️ shared objects, `OperationDim`, overrides | as R6, R7 |
| R19 | Running a graph is at least as fast as today; nesting adds no cost. | ⚠️ cost grows with each nesting level | Benchmark: run time of an FCN in 0, 2, 4 wrapping GraphNodes is constant and not above `main` (spike: 111–116 µs vs. 156–187 µs). |
| R20 | A size that fixes how a node is built (e.g. input neurons, branch input size) is a HyperParameter. Left open, it is inferred from the connected data, the same at all places of the node; a tuner changing it changes the configurations that depend on it. Sizes that depend on the place never flow into the state of a node. | ❌ four nodes derive state in `update_data_configs`; HyperParameters are unrelated to configurations | *`test_matrix_hyperparameters`*: models (FCN, DeepONet, CNN encoding) × origin (fixed, tuned, inferred) × places (one, two agreeing, two disagreeing → error at connecting) |

Out of scope: checking real tensor shapes against configurations at run
time (P2); addressing axes by meaning (`Mean(over=t)`).
