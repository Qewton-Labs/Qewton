from pioneer.algorithms.implementation import DEFAULT_DL_IMPLEMENTATION

from ..base import OperationNode
from ..implementation import (
    TorchImplementation,
)
from ...graphs.nodes import NO_DEFAULT


class Narrow(OperationNode):
    args = {"x": NO_DEFAULT, "dim": NO_DEFAULT, "start": 0, "length": NO_DEFAULT}
    outputs = ["output"]
    # TODO : Add Tensorflow implementation
    implementations = {TorchImplementation: ("narrow",)}

    def __init__(self, dim=None, start=0, length=None, backend=DEFAULT_DL_IMPLEMENTATION):
        self.args = self.args.copy()
        self.args["dim"] = dim if dim is not None else NO_DEFAULT
        self.args["start"] = start
        self.args["length"] = length if length is not None else NO_DEFAULT
        super().__init__(name=None, backend=backend)
