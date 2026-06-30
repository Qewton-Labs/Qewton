from pathlib import Path
import gmsh


def mesh_geometry(
    input_file: str | Path,
    output_file: str | Path,
    dim: int = 3,
    mesh_size: float | None = None,
):
    """Mesh a CAD geometry using Gmsh.

    Args:
        input_file (str | Path): Path to a STEP/STL/BREP/IGES/... geometry.
        output_file (str | Path): Path for the output mesh.
        dim (int, optional): The dimension of the object. Defaults to 3.
        mesh_size (float | None, optional): The maximum edge size in the mesh.
            Defaults to None.
    """
    gmsh.initialize()
    gmsh.model.add("model")

    input_file = str(input_file)

    if input_file.lower().endswith(".stl"):
        # STL is already a surface mesh
        gmsh.merge(input_file)

        # Create geometry from the discrete mesh
        gmsh.model.mesh.classifySurfaces(
            angle=40 * 3.14159 / 180,
            boundary=True,
            forReparametrization=True,
        )
        gmsh.model.mesh.createGeometry()

    else:
        # STEP, BREP, IGES, ...
        gmsh.model.occ.importShapes(input_file)
        gmsh.model.occ.synchronize()

    if mesh_size is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    gmsh.model.mesh.generate(dim)

    gmsh.write(str(output_file))

    gmsh.finalize()
