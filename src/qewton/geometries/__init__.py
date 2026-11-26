from .base import Geometry, DiscreteGeometry, BoundaryGeometry

from .continuous import *
from .discrete import *

try:
    from .cad.create_mesh import create_mesh_geometry

except (ImportError, AttributeError, OSError) as _err:
    # Gmsh is not installed, or its shared library cannot be loaded (OSError,
    # e.g. libGLU missing on headless Linux machines). Fail only when used.
    _gmsh_import_error = _err

    def create_mesh_geometry(*args, **kwargs):  # pylint: disable=unused-argument
        """Placeholder, since Gmsh could not be imported."""
        raise ImportError(
            f"create_mesh_geometry requires Gmsh, which could not be imported "
            f"({_gmsh_import_error}). Install it via pip install 'qewton[gmsh]'. "
            "On Linux, Gmsh also needs some system libraries, e.g. libGLU "
            "(apt install libglu1-mesa)."
        ) from _gmsh_import_error
