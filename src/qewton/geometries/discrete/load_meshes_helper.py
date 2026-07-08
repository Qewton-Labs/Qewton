# This file contains different methods that use gmsh to load different
# mesh formats. The formats have differences in how we have to read out
# markers and domain information.
import meshio

from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.config.dtypes import Int32


def load_file_with_meshio(
    file_path,
    marker_key: str | None = None,
    default_cell_tags: int = -1,
    backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
):
    msh = meshio.read(file_path)
    # Read all cell data:
    priority = ["tetra", "triangle", "line", "vertex"]
    p_key, key_idx = "", 0
    for key_idx, p_key in enumerate(priority):
        if p_key in msh.cells_dict:
            break
    # Check for markers of the cells and facets.
    cells, cell_markers, faces, face_markers = read_markers_from_file(
        msh, marker_key, default_cell_tags, p_key, priority[key_idx + 1], backend
    )
    return msh.points, cells, cell_markers, faces, face_markers, msh.field_data


def read_markers_from_file(
    msh,
    marker_key,
    default_cell_tags,
    cell_key,
    face_key,
    backend: type[ComputingBackend[TensorType]],
):
    # First try to find a key if not provided by the user
    if marker_key is None:
        candidates = [
            "gmsh:physical",
            "material",
            "Material",
            "region",
            "cell_tags",
        ]
        for c in candidates:
            if c in msh.cell_data:
                marker_key = c
                break
    # build all cells:
    cells = []
    for block in msh.cells:
        if block.type == cell_key:
            cells.extend(block.data)
    faces = []
    cell_markers = default_cell_tags * backend.math.ones(
        (len(cells),), dtype=backend.dtypes[Int32]
    )
    face_markers = backend.math.empty((0,), dtype=backend.dtypes[Int32])
    # Try to read out the markers from the mesh:
    if marker_key in msh.cell_data:
        marker_counter = 0
        for i, block in enumerate(msh.cells):
            # Cell data has markers for each block
            markers = backend.build_tensor(
                msh.cell_data[marker_key][i], dtype=backend.dtypes[Int32]
            )
            # Markers of main dimension:
            if block.type == cell_key:
                assert not any(
                    m == default_cell_tags for m in markers
                ), f"Default marker {default_cell_tags} tag found in the mesh \
                    data. This can lead to unexpected behavior."
                cell_markers[marker_counter : marker_counter + len(markers)] = markers
                marker_counter += len(markers)
            # Facets marker:
            elif block.type == face_key:
                faces.extend(block.data)
                face_markers = backend.math.concatenate((face_markers, markers))
    elif marker_key is not None:
        print(f"Could not find cell marker information. Mesh contains \
            {msh.cell_dict.keys()} which does not have the key {marker_key}.")
    return cells, cell_markers, faces, face_markers
