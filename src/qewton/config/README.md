# Config Submodule

Describes the data passed between nodes: its axes, their sizes and its
dtype. Graphs match these descriptions when ports are connected, so
mismatches are found while building and sizes are inferred from the
connected data.

```python
DataConfiguration(BatchAxes(...), GeometryAxes(grid), FeatureAxes(Variable("f", 1)))
```

describes a batch of fields `f` on the points of `grid`.

## Configurations and axes

- **`DataConfiguration`**: a sequence of axes plus `dtype` and
  `element_dtype`; at most one `FeatureAxes`.
- **Axes**: `BatchAxes` (samples), `GeometryAxes` (points of a `Geometry`),
  `FeatureAxes` (values, named by `Variable`s), `EllipsisAxes` (any sequence
  of axes, possibly none).
- **Dims**: `AxesDim(size)` with `size` an int, None (unknown) or a
  `HyperParameter`; `EllipsisDim` for any sequence of dims.

## Matching

Declared configurations never change. A `Matching` holds what matching
finds, e.g. for one graph:

- `matching.place(key, {port: declaration})` instantiates declarations at a
  place: open objects (unknown dims, ellipses and what contains them) get a
  stand-in there, every other object is used as it is;
- `config.merge(other)` (or `matching.merge(a, b)` for axes and dims) makes
  both describe the same data: corresponding axes and dims become one
  group, an ellipsis is bound to the part it stands for, the dtypes are
  merged;
- `config.resolve()` gives the configuration as known so far: every object
  replaced by the representative of its group, equal axes as one object.

Configurations of a matching are `DynamicDataConfiguration`s (a
`DataConfiguration` with its `matching`): only they have `merge`,
`merge_dim`, `resolve` and the shape operations, and the results of those
belong to the same matching. A contradiction raises
`DataConfigMismatchError`, naming the ports the conflicting values come
from. Graphs match along their edges (see `qewton.graphs`).

## Relations of nodes

A node relates its ports in the annotations of `forward`: one object used
in several annotations is the same at every place of the node.

```python
ellipsis = EllipsisAxes()

def forward(self, x: Number[TensorType, DataConfiguration(ellipsis)]
            ) -> Number[TensorType, DataConfiguration(ellipsis)]:
```

A node that changes dims by their index also implements `relate`, built
from the shape operations of the `DynamicDataConfiguration`s it gets: `drop`, `squeeze`,
`set_size`, `unsqueeze`, `permute`, `flatten`, `unflatten`, `reshape`,
`concat`, `split`, `broadcast`, `window`.

```python
def relate(self, inp: DataConfiguration) -> DataConfiguration:
    return inp.flatten(self.start_dim, self.end_dim)
```

They reuse the input's objects and compute sizes with operation dims
(`AddedDim`, `ProductDim`, `SequenceProductDim`, `WindowDim`,
`BroadcastDim`, …), which solve for an operand where the answer is unique.
`broadcast` makes size 1 and N compatible without merging them.

## HyperParameters in configurations

A size that fixes how a node is built is a `HyperParameter`; the dims of
that size are bound to it with `AxesDim(hp)` and follow its value.
`HyperParameter.size_from(x, name)` takes a given size, or None for one
that is inferred from the connected data (`HyperParameter.inferred`). An
inferred size is the same at all places of the node; places that disagree
raise at connecting. Arithmetic on HyperParameters (`hp * 2`) gives a
`HyperParameterOperation`, which is read but not inferred.

## Sizes of data sources

The outputs of a `DataLoader` or `PointSampler` share one batch axes, bound
to its `batch_size` or `n_points`. Changing them, or switching a sampler to
mesh mode or an active discretization, changes every dim behind it without
matching. The size at run time is not checked.

## dtypes

`dtype` is `Any`, a Python type, `TensorType`, the tensor class of a
backend, or a union of these; `element_dtype` is open, `Number` or a marker
such as `Float32` or `Complex64`, given in annotations as
`Complex64[TensorType, config]`. Two dtypes match if they have a common
member; the more special one is kept.

## Saving

Static configurations are saved with their dtypes; matched configurations
are rebuilt when the loaded graph is matched.

See [REQUIREMENTS.md](REQUIREMENTS.md) and [DECISIONS.md](DECISIONS.md).
