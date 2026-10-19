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
