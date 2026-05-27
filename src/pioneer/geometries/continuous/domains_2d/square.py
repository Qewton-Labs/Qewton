from numpy import ndarray

from pioneer.config.variables import Variable

from .parallelogram import Parallelogram


class Square(Parallelogram):

    def __init__(
        self,
        variable: Variable,
        origin: ndarray | list[float] | tuple[float, float],
        width: float,
        height: float,
    ):
        corner_1 = [origin[0] + width, origin[1]]
        corner_2 = [origin[0], origin[1] + height]
        super().__init__(variable, origin, corner_1, corner_2)
