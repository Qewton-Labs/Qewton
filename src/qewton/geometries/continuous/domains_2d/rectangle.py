from numpy import ndarray

from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_2d.parallelogram import Parallelogram


class Rectangle(Parallelogram):
    """Class for rectangles represented by origin, width and height.

    Args:
        variable (Variable): The variable associated with the rectangle, must be 2D.
        origin (np.ndarray | list[float] | tuple[float, float]):
            The origin of the rectangle (lower left corner).
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
    """

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
