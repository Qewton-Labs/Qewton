import itertools


def mean_edge_length(backend, mesh) -> float:
    """Mean length of a mesh's unique edges - used as the natural resolution
    unit for a grid resampling that mesh's field: linear interpolation
    inside cells is continuous but its derivatives jump at cell boundaries,
    so a grid much finer than the mesh would make those boundaries visible
    and look like a rendering artifact. Shared by PlaneSliceGeometry (2D
    plane-stack default resolution) and VolumeGridGeometry (3D box default
    resolution) - identical need, only the grid shape built around it
    differs."""
    vertices = mesh.vertices
    cells = mesh.cells
    pairs = list(itertools.combinations(range(cells.shape[1]), 2))
    edges = backend.math.concatenate(
        [backend.math.sort(cells[:, [a, b]], axis=1) for a, b in pairs], axis=0
    )
    edges = backend.math.unique(edges, axis=0)
    lengths = backend.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
    )
    return float(backend.math.mean(lengths))
