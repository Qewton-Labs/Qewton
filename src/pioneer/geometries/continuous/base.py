from types import EllipsisType

from ...config.variables import Variable
from ..base import Geometry, BoundaryGeometry


class ContinuousGeometry(Geometry):

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

    def __add__(self, other):
        from .domain_operations.union import UnionGeometry

        assert isinstance(other, ContinuousBoundaryGeometry)
        return UnionGeometry(self.geometry, other.geometry).boundary

    def __sub__(self, other):
        from .domain_operations.cut import CutGeometry

        assert isinstance(other, ContinuousBoundaryGeometry)
        return CutGeometry(self.geometry, other.geometry).boundary

    def __and__(self, other):
        from .domain_operations.intersection import IntersectionGeometry

        assert isinstance(other, ContinuousBoundaryGeometry)
        return IntersectionGeometry(self.geometry, other.geometry).boundary
