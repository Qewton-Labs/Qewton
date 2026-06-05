from types import EllipsisType

from qewton.config.variables import Variable
from qewton.geometries.base import Geometry, BoundaryGeometry


class ContinuousGeometry(Geometry):
    """The parent class for all built-in continuous geometries.
    Can be used just like any other Geometry. They are allow for
    operations like union, intersection and cut with other continuous geometries.
    The are essentially Constructive Solid Geometry (CSG) objects, only defined
    by a small amount of parameters.

    A continuous geometry can be discretized by creating a mesh.
    """

    def __init__(
        self,
        variable: Variable,
        shape: tuple[int | None, ...] | EllipsisType = ...,
    ):
        super().__init__(variable, dim=variable.dim, shape=shape)

    def create_boundary(self):
        return ContinuousBoundaryGeometry(self)

    def __add__(self, other):
        from .domain_operations.union import UnionGeometry

        assert isinstance(other, ContinuousGeometry)
        return UnionGeometry(self, other)

    def __sub__(self, other):
        from .domain_operations.cut import CutGeometry

        assert isinstance(other, ContinuousGeometry)
        return CutGeometry(self, other)

    def __and__(self, other):
        from .domain_operations.intersection import IntersectionGeometry

        assert isinstance(other, ContinuousGeometry)
        return IntersectionGeometry(self, other)


class ContinuousBoundaryGeometry(BoundaryGeometry):

    def __init__(self, geometry):
        assert isinstance(geometry, ContinuousGeometry)
        super().__init__(geometry)
        self.geometry: ContinuousGeometry = geometry

    def create_mesh(self, max_vertex_distance: float | None = None):
        return self.geometry.create_mesh(max_vertex_distance=max_vertex_distance).boundary

    def __add__(self, other):
        assert isinstance(other, ContinuousBoundaryGeometry)
        return (self.geometry + other.geometry).boundary

    def __sub__(self, other):
        assert isinstance(other, ContinuousBoundaryGeometry)
        return (self.geometry - other.geometry).boundary

    def __and__(self, other):
        assert isinstance(other, ContinuousBoundaryGeometry)
        return (self.geometry & other.geometry).boundary
