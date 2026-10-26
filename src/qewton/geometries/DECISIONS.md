# Geometries: Design Decisions

## Product geometries are meshed factor by factor
`ProductGeometry.create_mesh` meshes each factor separately and combines the
factor meshes into a `ProductMeshGeometry`. The product vertices are ordered
with the first factor varying slowest, so the vertex axis can be reshaped to
one axis per factor. The full simplex mesh is triangulated (staircase
triangulation, factor cell vertices sorted by global index for a conforming
mesh) only when `mesh` is accessed, since many uses only need the vertices.

## `SampledGeometry` is shared, not copied
`SampledGeometry.__deepcopy__` returns the object itself. Data configurations
are deep-copied (e.g. a node's dynamic configuration from its static one),
and every copy must keep referencing the object that the owning sampler
updates in each forward pass.

## Mesh availability is checked via the `mesh` attribute
Code that needs a mesh (e.g. plotting) checks `getattr(geometry, "mesh", None) is not None`
instead of the geometry's class. A `SampledGeometry` exposes a mesh only when
its current points have cell connectivity, and consumers do not need to know
about it.

## Geometries have an explicit device, like nodes
A geometry can keep its data (e.g. mesh vertices, a center or corners) on a
GPU, so that `contains` or sampling many points runs there without
transfers. `geometry.device` tells where the data is, and only an explicit
`to(device)` changes it, never another method. `to()` also moves related
geometries: a boundary and its domain, the factors of a product, the parts of
a union, cut or intersection. Nodes owning a geometry move it in their own
`to()` (a `PointSampler`, a `MeshInterpolationNode`). Copying meshes per
device instead would duplicate data and break identity checks, e.g. of a
`MeshGeometry` in a data configuration.

Methods called with points or a `device` on another device than the
geometry still compute correctly, without changing the geometry:
- Continuous geometries bring their few constants (corners, center, ...) to
  the points' device and compute there.
- Discrete geometries (meshes, grids, point clouds) compute on their own
  device: points are brought there and the result is moved back. Since that
  transfer can be costly, it issues a `DeviceTransferWarning` suggesting
  `to(device)`.

## Samplers mesh their own geometry
A `PointSampler` meshes its geometry for visualization and caches the mesh
per resolution and device. The `SampledGeometry` it updates only holds the
current points and is shared between copies of the sampler (see above), so it
keeps no reference to a geometry: a deep copy of a sampler, e.g. one tuning
trial, meshes and moves only its own geometry.

## `create_mesh(device)` creates new data on `device`
Every geometry that creates new mesh data creates it on the requested
device. An already discrete geometry (e.g. a loaded `MeshGeometry`) returns
itself unchanged instead of a copy. Code that needs the points on a device
moves the points it hands on, e.g. a `PointSampler` in mesh mode moves the
mesh vertices to its mesh device.
