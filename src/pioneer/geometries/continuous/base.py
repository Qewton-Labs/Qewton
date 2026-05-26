from types import EllipsisType

from ...config.variables import Variable
from ..base import Geometry


class ContinuousGeometry(Geometry):

    def __init__(
        self,
        variable: Variable,
        shape: tuple[int | None, ...] | EllipsisType = ...,
    ):
        super().__init__(variable, dim=variable.dim, shape=shape)

    def __add__(self, other):
        from .domain_operations.union import UnionGeometry

        assert isinstance(other, ContinuousGeometry)
        return UnionGeometry(self, other)
