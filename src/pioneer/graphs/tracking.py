from ..algorithms.building_blocks import (
    Add,
    MatMul,
    Subtract,
    Multiply,
    Power,
    Divide,
    Abs,
)
from .graphs import Graph
from .nodes import OutputPort


class TrackingObject:
    current_graph_tracked: Graph | None = None

    def __init__(self, last_output_port: OutputPort | None = None):
        self.last_output_port: OutputPort | None = last_output_port

    def __add__(self, other):
        add_node = Add()
        return add_node(self, other)

    def __matmul__(self, other):
        matmul_node = MatMul()
        return matmul_node(self, other)

    def __sub__(self, other):
        subtract_node = Subtract()
        return subtract_node(self, other)

    def __mul__(self, other):
        multiply_node = Multiply()
        return multiply_node(self, other)

    def __pow__(self, other):
        power_node = Power()
        return power_node(self, other)

    def __truediv__(self, other):
        divide_node = Divide()
        return divide_node(self, other)

    def __abs__(self):
        abs_node = Abs()
        return abs_node(self)
