from .base import Geometry, DiscreteGeometry, BoundaryGeometry

from .continuous import *
from .discrete import *

try:
    from .cad.create_mesh import create_mesh_geometry

except (ImportError, AttributeError):
    # Gmsh is not installed
    pass
