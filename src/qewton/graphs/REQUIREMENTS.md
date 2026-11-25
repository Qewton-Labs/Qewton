# Graphs: Requirements

What graphs must achieve for matching and running (Part 1) and the
requirements on each component (Part 2). [README.md](README.md) and
[DECISIONS.md](DECISIONS.md) follow the same components. Merging of
configurations is in [config/REQUIREMENTS.md](../config/REQUIREMENTS.md).

# Part 1: General requirements

## Scenarios

| ID | Scenario |
|---|---|
| S1 | PINN: one model is used in several pipelines (PDE, boundary, initial condition) with different samplers and batch sizes; the weights are shared. One sampler can feed several pipelines. |
| S2 | Weight sharing in one graph: a model evaluated at several places, e.g. called twice in a tracked function. |
| S3 | A GraphNode whose inner graph depends on sizes inferred from the connected data, e.g. a DeepONet. |
| S4 | Inspect the configurations inside nested graphs, e.g. with `graph.diagram`. |

## Requirements

| ID | Requirement | Scenarios |
|---|---|---|
| G1 | A node can be used in several graphs and at several places of one graph. | S1, S2 |
| G2 | Deeply nested GraphNodes are the normal case; running them adds no cost from ports or nesting. | all |
| G3 | Building grows linearly with the size of a graph and the depth of nesting. | all |

# Part 2: Architecture

A *place* is a node at a path of GraphNodes within a graph. A graph keeps
the matched configuration and the value of every port per place; nodes
keep only what is shared by all places (weights, hyperparameters).

Status: done, partial (with the gap named), open.

## Places (`graphs.py`)

| ID | Requirement | From | Status |
|---|---|---|---|
| PL1 | Configurations and values are kept per place; what a node declares open (e.g. features without a Variable) becomes concrete per place. Weights and other state of a node are shared by all its places. | G1 | done |
| PL2 | A GraphNode in several graphs is matched with each of them; the configurations inside it can be inspected per place. | G1, S4 | done |
| PL3 | Matching derives no state of a node. | G1 | partial: `Slice.relate` keeps the slice of its place (`slice_config`) |

## GraphNodes (`control_nodes/graph_node.py`)

| ID | Requirement | From | Status |
|---|---|---|---|
| GN1 | A graph matches and runs the inner graphs of its GraphNodes in place; running is one flat list of steps. | G2 | done |
| GN2 | A GraphNode with sizes still open has no inner graph until it is built; called on its own before, it raises a clear error. | S3 | done |

## Setup (`nodes.py`, `graphs.py`)

| ID | Requirement | From | Status |
|---|---|---|---|
| SU1 | Removing or replacing nodes and edges, and inner graphs rebuilt by `build`, leave consistent configurations and a valid order. | G1 | done |
| SU2 | Node authors implement `build`; `setup` belongs to the library and builds a node only once its sizes are known. | S3 | done |
| SU3 | The number of merges grows linearly with the length of a chain and with the depth of nesting. | G3 | done |
