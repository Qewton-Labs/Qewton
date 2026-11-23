# Config: Requirements

What data configurations and their matching must achieve (Part 1) and the
requirements on each component (Part 2). [README.md](README.md) and
[DECISIONS.md](DECISIONS.md) follow the same components. Places, GraphNodes
and setup are in [graphs/REQUIREMENTS.md](../graphs/REQUIREMENTS.md).

# Part 1: General requirements

## Scenarios

| ID | Scenario |
|---|---|
| S1 | A model is connected to data whose sizes it does not know beforehand, e.g. the input size of an FCN or the branch size of a DeepONet. |
| S2 | Plot a port with reference data: prediction, reference and error share one control per batch axis. |
| S3 | Plot in mesh mode: samplers provide another number of points than during training. |
| S4 | Tuning: hyperparameters become concrete late; `setup()` rebuilds the node, possibly with other port configurations. |
| S5 | Data pipelines: data loader → processing (normalization, PCA) → model, with different batch sizes for training, validation and test. |
| S6 | External developers write new nodes. |

## Requirements

| ID | Requirement | Scenarios |
|---|---|---|
| G1 | Connecting two ports merges their configurations: equal axes and equal dims are one object, compared with `is`. | S2, S3 |
| G2 | All matching happens while building or setting up a graph; running has no overhead from configurations. | all |
| G3 | Few base rules; convenience comes from structures built on them, not from special cases. | S6 |
| G4 | The result of matching does not depend on the order of `connect` calls. | all |
| G5 | The rules a node author must know are few and local to the node. | S6 |
| G6 | A contradiction is an error at connecting that names where the conflicting values come from. | all |

# Part 2: Architecture

A `DataConfiguration` is a sequence of axes (`BatchAxes`, `GeometryAxes`,
`FeatureAxes`, `EllipsisAxes`), each a sequence of dims (`AxesDim`), plus a
`dtype` and an `element_dtype`. Nodes declare configurations with unknowns
in the annotations of `forward`; a graph instantiates them per place and
merges them along its edges. A dim may be computed from others
(`OperationDim`) or bound to a HyperParameter.

Status: done, partial (with the gap named), open.

## Merging (`axes.py`, `sequences.py`, `data_configurations/`)

| ID | Requirement | From | Status |
|---|---|---|---|
| M1 | Merging makes corresponding axes and dims one object; a size known at one place holds at every place merged with it, also inside GraphNodes. Same axes = semantically the same axis; same dim = same size only. | G1 | done |
| M2 | A node passes axes through by using one object in several of its annotations; objects shared within a declaration stay shared in every instance of it. | G1, G5 | done |
| M3 | An ellipsis binds to a sequence of axes or dims, possibly empty; a shared ellipsis stands for the same sequence. An ambiguous alignment waits until more is known. | G4 | done |
| M4 | Merging never changes Geometries, Variables or the static configurations of ports. | G1 | done |
| M5 | A contradiction names both origins of the conflicting values. | G6 | done |

## Relations of nodes (`Node.relate`, shape operations, `OperationDim`)

| ID | Requirement | From | Status |
|---|---|---|---|
| N1 | A node declares the relations between its ports only in its annotations and, if it changes dims by their index, in a pure function `relate` built from the shape operations of `DataConfiguration`. | G3, G5 | done |
| N2 | Sizes may depend arithmetically on others (sum, product, window, broadcasting); they are resolved in every direction where unique. | G1 | partial: a dim solved backwards from a size bound to a HyperParameter keeps the number it was solved to |
| N3 | Broadcasting (size 1 against N) makes dims compatible without merging them; element-wise nodes with several inputs relate them by broadcasting. | G1 | done |

## HyperParameters in configurations (`AxesDim(hp)`, `HyperParameter`)

| ID | Requirement | From | Status |
|---|---|---|---|
| H1 | A size that fixes how a node is built is a HyperParameter, and the dims of that size are bound to it. | S1, S4 | done |
| H2 | An input size left open (None) is inferred from the connected data, the same at all places of the node; places that disagree are an error at connecting. | S1, G6 | done |
| H3 | A tuner changing a HyperParameter changes every dim bound to it. | S4 | done |
| H4 | Sizes that depend on the place (batch, number of points) never flow into the state of a node. | S1, S5 | done |

## Sizes of data sources (`DataLoader`, `PointSampler`)

| ID | Requirement | From | Status |
|---|---|---|---|
| D1 | The outputs of a data source share one batch axes. | G1, S2, S5 | done |
| D2 | Batch and point dims are bound to the HyperParameters of their source (`batch_size`, `n_points`); changing them changes every dim behind them at once, without matching. Mesh mode and an active discretization set `n_points` to the number of points provided. | G2, S3, S5 | done |
| D3 | A batch dim has the configured size; the actual size at run time (e.g. a smaller last batch) is neither checked nor written back. | G2 | done |

## dtypes (`dtypes.py`)

| ID | Requirement | From | Status |
|---|---|---|---|
| T1 | `dtype` (`Any`, a Python type, `TensorType`, a backend tensor class, or a union of these) and `element_dtype` (`Number`, `Float32`, …) are matched including subtypes; the more special one wins. | G1, G6 | done |

## Saving (`saving/`)

| ID | Requirement | From | Status |
|---|---|---|---|
| V1 | Static configurations are saved with their dtypes and loaded without importing classes outside `qewton`; matched configurations are rebuilt after loading. | G1 | done |
| V2 | Models saved in the previous file format (version 1) load and compute as before. | G1 | partial: DeepONets of version 1 are not loaded |
