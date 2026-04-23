from pioneer.algorithms.implementation import DEFAULT_DL_IMPLEMENTATION

from ..base import OperationNode
from ..implementation import (
    TorchImplementation,
    TensorflowImplementation,
)
from ...graphs.nodes import NO_DEFAULT

# The following classes represent basic mathematical operations.
# They are designed to work with different operations and one only needs to pass
# in the name of the respective function in the backend to create a new operation.
# This reduces the number of classes we need to create. However, if there are
# specific operations that require special handling, we can of course
# create separate classes for them.


# region: Arithmetic operations


class Add(OperationNode):
    args = {"x": NO_DEFAULT, "y": NO_DEFAULT}
    outputs = ["output"]
    implementations = {TorchImplementation: ("add",), TensorflowImplementation: ("add",)}


class Subtract(OperationNode):
    args = {"x": NO_DEFAULT, "y": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("sub",),
        TensorflowImplementation: ("subtract",),
    }


class Multiply(OperationNode):
    args = {"x": NO_DEFAULT, "y": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("mul",),
        TensorflowImplementation: ("multiply",),
    }


class Divide(OperationNode):
    args = {"x": NO_DEFAULT, "y": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("true_divide",),
        TensorflowImplementation: ("truediv",),
    }


class Mod(OperationNode):
    args = {"x": NO_DEFAULT, "y": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("remainder",),
        TensorflowImplementation: ("mod",),
    }


# endregion


# region: Powers and roots


class Square(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("square",),
        TensorflowImplementation: ("square",),
    }


class Sqrt(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("sqrt",),
        TensorflowImplementation: ("sqrt",),
    }


class Power(OperationNode):
    args = {"x": NO_DEFAULT, "y": NO_DEFAULT}
    outputs = ["output"]
    implementations = {TorchImplementation: ("pow",), TensorflowImplementation: ("pow",)}


# endregion


# region: Exponential and logarithmic functions


class Exp(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {TorchImplementation: ("exp",), TensorflowImplementation: ("exp",)}


class Log(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {TorchImplementation: ("log",), TensorflowImplementation: ("log",)}


class Log2(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("log2",),
        TensorflowImplementation: ("keras.ops.log2",),
    }


class Log10(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("log10",),
        TensorflowImplementation: ("log10",),
    }


# endregion


# region: Trigonometric functions


class Sin(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {TorchImplementation: ("sin",), TensorflowImplementation: ("sin",)}


class Cos(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {TorchImplementation: ("cos",), TensorflowImplementation: ("cos",)}


class Tan(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {TorchImplementation: ("tan",), TensorflowImplementation: ("tan",)}


class ArcSin(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("arcsin",),
        TensorflowImplementation: ("asin",),
    }


class ArcCos(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("arccos",),
        TensorflowImplementation: ("acos",),
    }


class ArcTan(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("arctan",),
        TensorflowImplementation: ("atan",),
    }


# endregion


# region: Other useful math functions


class Abs(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {TorchImplementation: ("abs",), TensorflowImplementation: ("abs",)}


class Floor(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("floor",),
        TensorflowImplementation: ("floor",),
    }


class Ceil(OperationNode):
    args = {"x": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("ceil",),
        TensorflowImplementation: ("ceil",),
    }


class Maximum(OperationNode):
    args = {"x": NO_DEFAULT, "y": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("maximum",),
        TensorflowImplementation: ("maximum",),
    }


class Minimum(OperationNode):
    args = {"x": NO_DEFAULT, "y": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("minimum",),
        TensorflowImplementation: ("minimum",),
    }


# endregion


# region: Matrix operations


class MatMul(OperationNode):
    args = {"x": NO_DEFAULT, "y": NO_DEFAULT}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("matmul",),
        TensorflowImplementation: ("matmul",),
    }


class SVD(OperationNode):

    args = {"x": NO_DEFAULT}
    outputs = ["U", "S", "V"]
    implementations = {
        TorchImplementation: ("svd",),
        TensorflowImplementation: ("linalg.svd",),
    }


# endregion


# region: Statistic operations


class Mean(OperationNode):
    """
    Computes the mean of the input tensor along the specified axis.

    # TODO: how to write docstrings for these methods?
    """

    args = {"x": NO_DEFAULT, "axis": NO_DEFAULT, "keepdims": False}
    outputs = ["output"]
    # Orders should be correct
    implementations = {
        TorchImplementation: ("mean",),
        TensorflowImplementation: ("reduce_mean",),
    }

    def __init__(
        self,
        axis: type[NO_DEFAULT] | None | int | tuple[int] = NO_DEFAULT,
        keepdims=False,
        backend=DEFAULT_DL_IMPLEMENTATION,
    ):
        self.args = self.args.copy()
        self.args["axis"] = axis
        self.args["keepdims"] = keepdims
        super().__init__(name=None, backend=backend)


class Std(OperationNode):

    args = {"x": NO_DEFAULT, "axis": NO_DEFAULT, "keepdims": False}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("std",),
        TensorflowImplementation: ("reduce_std",),
    }

    def __init__(
        self,
        axis: type[NO_DEFAULT] | None | int | tuple[int] = NO_DEFAULT,
        keepdims=False,
        backend=DEFAULT_DL_IMPLEMENTATION,
    ):
        self.args = self.args.copy()
        if isinstance(axis, int):
            axis = (axis,)
        self.args["axis"] = axis
        self.args["keepdims"] = keepdims
        super().__init__(name=None, backend=backend)


# endregion


# region: Reshaping operations


class Flatten(OperationNode):
    args = {"x": NO_DEFAULT, "start_dim": 0, "end_dim": -1}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("flatten",),
        TensorflowImplementation: ("flatten",),
    }

    def __init__(
        self,
        start_dim: int = 0,
        end_dim: int = -1,
        backend=DEFAULT_DL_IMPLEMENTATION,
    ):
        self.args = self.args.copy()
        self.args["start_dim"] = start_dim
        self.args["end_dim"] = end_dim
        super().__init__(name=None, backend=backend)


class Transpose(OperationNode):
    args = {"x": NO_DEFAULT, "perm": [1, 0]}
    outputs = ["output"]
    implementations = {
        TorchImplementation: ("permute",),
        TensorflowImplementation: ("transpose",),
    }

    def __init__(
        self,
        perm: list | None = None,
        backend=DEFAULT_DL_IMPLEMENTATION,
    ):
        self.args = self.args.copy()
        self.args["perm"] = perm if perm is not None else [1, 0]
        super().__init__(name=None, backend=backend)


# endregion
