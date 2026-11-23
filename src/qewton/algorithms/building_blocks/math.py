from typing import Annotated

from qewton.backends import DEFAULT_DL_BACKEND, TensorType, DeepLearningBackend
from qewton.backends.base import Backend
from qewton.graphs.nodes import Node, NodeState
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.config.errors import StructureUnknown
from qewton.config.axes import (
    EllipsisAxes,
    AxesDim,
    EllipsisDim,
    FeatureAxes,
    MinimumDim,
)

# The following classes represent basic mathematical operations.
# They are designed to work with different operations and one only needs to pass
# in the name of the respective function in the backend to create a new operation.
# This reduces the number of classes we need to create. However, if there are
# specific operations that require special handling, we can of course
# create separate classes for them.


# region: Arithmetic operations


class BroadcastingNode(Node[TensorType]):
    """A node computing element by element from two inputs that broadcast
    against each other: where one has size 1, the other's size is taken.
    Subclasses declare each port with an ellipsis of its own; an input may
    also be a Python scalar, which the backends convert themselves."""

    def relate(self, x: DC, y: DC) -> DC:
        return x.broadcast(y)


class Add(BroadcastingNode[TensorType]):
    """A summation node that adds two input tensors together.
    Has two input ports and one output port.
    """

    def forward(
        self,
        x: Annotated[TensorType | float, DC(EllipsisAxes())],
        y: Annotated[TensorType | float, DC(EllipsisAxes())],
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.add(x, y)


class Subtract(BroadcastingNode[TensorType]):
    """A subtraction node that subtracts the second input from the first input tensor.
    Has two input ports and one output port.
    """

    def forward(
        self,
        x: Annotated[TensorType | float, DC(EllipsisAxes())],
        y: Annotated[TensorType | float, DC(EllipsisAxes())],
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.subtract(x, y)


class Multiply(BroadcastingNode[TensorType]):
    """Element-wise multiplication of two input tensors.
    Has two input ports and one output port.
    """

    def forward(
        self,
        x: Annotated[TensorType | float, DC(EllipsisAxes())],
        y: Annotated[TensorType | float, DC(EllipsisAxes())],
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.multiply(x, y)


class Divide(BroadcastingNode[TensorType]):
    """Element-wise division of the first input by the second input tensor.
    Has two input ports and one output port.
    """

    def forward(
        self,
        x: Annotated[TensorType | float, DC(EllipsisAxes())],
        y: Annotated[TensorType | float, DC(EllipsisAxes())],
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.divide(x, y)


class Mod(BroadcastingNode[TensorType]):
    """Element-wise modulo (remainder) operation on two input tensors.
    Has two input ports and one output port.
    """

    def forward(
        self,
        x: Annotated[TensorType | float, DC(EllipsisAxes())],
        y: Annotated[TensorType | float, DC(EllipsisAxes())],
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.mod(x, y)


class Negative(Node[TensorType]):
    """Negation node that returns the negative of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.negative(x)


# endregion


# region: Powers and roots


class Square(Node[TensorType]):
    """Element-wise squaring of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.square(x)


class Sqrt(Node[TensorType]):
    """Element-wise square root of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.sqrt(x)


class Power(BroadcastingNode[TensorType]):
    """Element-wise exponentiation node that raises the first input to the
    power of the second input. Has two input ports and one output port.
    """

    def __init__(
        self,
        power: float | None = None,
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(name, backend=backend)
        self.power = power
        if power is not None:
            self.input_ports[-1].default = power

    def forward(
        self,
        x: Annotated[TensorType | float, DC(EllipsisAxes())],
        y: Annotated[TensorType | float, DC(EllipsisAxes())],
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        out = self.backend.math.power(x, y)
        return self.backend.math.power(x, y)


# endregion


# region: Exponential and logarithmic functions


class Exp(Node[TensorType]):
    """Element-wise exponential (e^x) of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.exp(x)


class Log(Node[TensorType]):
    """Element-wise natural logarithm of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.log(x)


class Log2(Node[TensorType]):
    """Element-wise base-2 logarithm of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.log2(x)


class Log10(Node[TensorType]):
    """Element-wise base-10 logarithm of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.log10(x)


# endregion


# region: Trigonometric functions


class Sin(Node[TensorType]):
    """Element-wise sine of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.sin(x)


class Cos(Node[TensorType]):
    """Element-wise cosine of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.cos(x)


class Tan(Node[TensorType]):
    """Element-wise tangent of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.tan(x)


class ArcSin(Node[TensorType]):
    """Element-wise inverse sine (arcsine) of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.arcsin(x)


class ArcCos(Node[TensorType]):
    """Element-wise inverse cosine (arccosine) of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.arccos(x)


class ArcTan(Node[TensorType]):
    """Element-wise inverse tangent (arctangent) of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.arctan(x)


# endregion


# region: Other useful math functions


class Abs(Node[TensorType]):
    """Element-wise absolute value of the input tensor.
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.abs(x)


class Floor(Node[TensorType]):
    """Element-wise floor operation on the input tensor (rounds down).
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.floor(x)


class Ceil(Node[TensorType]):
    """Element-wise ceiling operation on the input tensor (rounds up).
    Has one input port and one output port.
    """

    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.ceil(x)


class Maximum(BroadcastingNode[TensorType]):
    """Element-wise maximum of two input tensors.
    Has two input ports and one output port.
    """

    def forward(
        self,
        x: Annotated[TensorType | float, DC(EllipsisAxes())],
        y: Annotated[TensorType | float, DC(EllipsisAxes())],
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.maximum(x, y)


class Minimum(BroadcastingNode[TensorType]):
    """Element-wise minimum of two input tensors.
    Has two input ports and one output port.
    """

    def forward(
        self,
        x: Annotated[TensorType | float, DC(EllipsisAxes())],
        y: Annotated[TensorType | float, DC(EllipsisAxes())],
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.minimum(x, y)


# endregion


# region: Matrix operations


class MatMul(Node[TensorType]):
    """Matrix multiplication node that multiplies two tensors along their matrix dimensions.
    Supports batch dimensions and ellipsis dimensions.
    Has two input ports and one output port.
    """

    ell_ax = EllipsisAxes()
    dim_1 = AxesDim(None)
    dim_ellipsis = EllipsisDim()
    dim_2 = AxesDim(None)

    def forward(
        self,
        x: Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_ellipsis, dim_1)))],
        y: Annotated[TensorType, DC(EllipsisAxes(), FeatureAxes(shape=(dim_1, dim_2)))],
    ) -> Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_ellipsis, dim_2)))]:
        return self.backend.math.matmul(x, y)


class SVD(Node[TensorType]):
    """Singular Value Decomposition node that decomposes an input matrix into U, S, V components.
    Has one input port and three output ports (U, S, V).
    """

    dim_1 = AxesDim(None)
    dim_2 = AxesDim(None)
    min_dim = MinimumDim(dim_1, dim_2)
    ell_ax = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_1, dim_2)))],
    ) -> tuple[
        Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_1, dim_1)))],
        Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(min_dim,)))],
        Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_2, dim_2)))],
    ]:
        return self.backend.linalg.svd(x)


class Dot(Node[TensorType]):
    """Dot product node that computes the inner product of two vectors.
    Has two input ports and one output port.
    """

    dim_1 = AxesDim(None)

    def forward(
        self,
        x: Annotated[TensorType, DC(EllipsisAxes(), FeatureAxes(shape=(dim_1,)))],
        y: Annotated[TensorType, DC(EllipsisAxes(), FeatureAxes(shape=(dim_1,)))],
    ) -> Annotated[TensorType, DC(EllipsisAxes(), FeatureAxes(shape=(1,)))]:
        return self.backend.math.inner(x, y)

    def relate(self, x: DC, y: DC) -> DC:
        return x.drop(-1).broadcast(y.drop(-1)).unsqueeze(-1)


class Norm(Node[TensorType]):
    """Norm node that computes the p-norm of the input tensor along the
    last axis. Always keeps the last dimension (keepdims=True).
    Has one input port and one output port.
    """

    dim_1 = AxesDim(None)
    ell_ax = EllipsisAxes()

    def __init__(
        self,
        ord: float | str = 2,
        name: str | None = None,
        state: NodeState = NodeState.FIXED,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        self.ord = ord
        super().__init__(name, state, backend, **kwargs)

    def forward(
        self,
        x: Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_1,)))],
    ) -> Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(1,)))]:
        return self.backend.linalg.norm(x, order=self.ord, axis=-1, keepdims=True)


class Inner(Node[TensorType]):
    """Inner product node that computes the element-wise product sum along the last axis.
    Supports keepdims option to retain dimension after reduction.
    Has two input ports and one output port.
    """

    dim_1 = AxesDim(None)
    ell_ax = EllipsisAxes()

    def __init__(
        self,
        keepdims: bool = False,
        name: str | None = None,
        state: NodeState = NodeState.FIXED,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        self.keepdims = keepdims
        super().__init__(name, state, backend, **kwargs)

    def forward(
        self,
        x: Annotated[TensorType, DC(EllipsisAxes(), FeatureAxes(shape=(dim_1,)))],
        y: Annotated[TensorType, DC(EllipsisAxes(), FeatureAxes(shape=(dim_1,)))],
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.sum(x * y, axis=-1, keepdims=self.keepdims)

    def relate(self, x: DC, y: DC) -> DC:
        joint = x.drop(-1).broadcast(y.drop(-1))
        return joint.unsqueeze(-1) if self.keepdims else joint


# endregion


# region: Statistic operations
class ReductionNode(Node[TensorType]):
    """Base class for reduction operations (Mean, Sum, Std) that reduce tensor dimensions.
    Supports axis specification and keepdims option.
    """

    def __init__(
        self,
        name=None,
        axis: None | int | tuple[int, ...] = None,
        keepdims=False,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(name=name, backend=backend)

    def relate(self, x: DC) -> DC:
        """Without keepdims the reduced dims are dropped, with keepdims they
        have size 1; without an axis every dim is reduced."""
        if self.axis is None:
            if not self.keepdims:
                return DC(dtype=x.dtype, element_dtype=x.element_dtype)
            if ... in x.resolve().shape:
                raise StructureUnknown(f"The dims of {x} are not known yet.")
            axes = tuple(range(len(x.shape)))
        else:
            axes = (self.axis,) if isinstance(self.axis, int) else tuple(self.axis)
        # Later indices first, so that the earlier ones stay valid.
        ordered = sorted(a for a in axes if a >= 0)[::-1] + sorted(a for a in axes if a < 0)
        result = x
        for axis in ordered:
            result = result.set_size(axis, 1) if self.keepdims else result.drop(axis)
        return result


class Mean(ReductionNode[TensorType]):
    """Mean reduction node that computes the average of input tensor along the specified axis.
    Has one input port and one output port.
    """

    def forward(
        self, x: Annotated[TensorType, DC(EllipsisAxes())]
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.mean(x, axis=self.axis, keepdims=self.keepdims)


class Sum(ReductionNode[TensorType]):
    """Sum reduction node that computes the total of input tensor elements along the specified axis.
    Has one input port and one output port.
    """

    def forward(
        self, x: Annotated[TensorType, DC(EllipsisAxes())]
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.sum(x, axis=self.axis, keepdims=self.keepdims)


class Std(ReductionNode[TensorType]):
    """Standard deviation reduction node that computes the standard deviation along the specified axis.
    Has one input port and one output port.
    """

    def forward(
        self, x: Annotated[TensorType, DC(EllipsisAxes())]
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.std(x, axis=self.axis, keepdims=self.keepdims)


# endregion


# region: Reshaping operations


class Transpose(Node[TensorType]):
    """Transpose node that permutes the dimensions of the input tensor.
    Has one input port and one output port.
    """

    def __init__(
        self,
        perm: list | None = None,
        backend=DEFAULT_DL_BACKEND,
    ):
        self.perm = perm if perm is not None else [1, 0]
        super().__init__(name=None, backend=backend)

    def relate(self, x: DC) -> DC:
        return x.permute(self.perm)

    def forward(
        self, x: Annotated[TensorType, DC.empty()]
    ) -> Annotated[TensorType, DC.empty()]:
        return self.backend.math.transpose(x, axes=self.perm)


# endregion


# endregion
