from ..base import Domain


class MeshDomain(Domain):

    def __init__(
        self, variable, mesh_vertices: list[list[float]], mesh_triangulation: list[list]
    ):
        assert (
            len(mesh_vertices[0]) == variable.dim
        ), "Dimension of variable must match dimension of mesh vertices."
        assert (
            len(mesh_triangulation[0]) == variable.dim + 1
        ), "Mesh triangulation must have dimension + 1 vertices."
        bounding_box = []
        for dim in range(variable.dim):
            dim_values = [vertex[dim] for vertex in mesh_vertices]
            bounding_box.append((min(dim_values), max(dim_values)))
        super().__init__(variable, bounding_box)
        self.mesh_vertices = mesh_vertices
        self.mesh_triangulation = mesh_triangulation
