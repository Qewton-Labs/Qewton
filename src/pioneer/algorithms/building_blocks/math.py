from typing import Annotated, Any

from ..backend import DEFAULT_DL_BACKEND
from ..base import BackendNode

from ...config.configuration_base import DataConfiguration
from ...graphs.nodes import NO_DEFAULT

# The following classes represent basic mathematical operations.
# They are designed to work with different operations and one only needs to pass
# in the name of the respective function in the backend to create a new operation.
# This reduces the number of classes we need to create. However, if there are
# specific operations that require special handling, we can of course
# create separate classes for them.


# region: Arithmetic operations


class Add(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
        y: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.add(x, y)


class Subtract(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
        y: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x, y)

    def torch_implementation(self, x, y):
        return self.backend.library.sub(x, y)

    def tensorflow_implementation(self, x, y):
        return self.backend.library.subtract(x, y)


class Multiply(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
        y: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x, y)

    def torch_implementation(self, x, y):
        return self.backend.library.mul(x, y)

    def tensorflow_implementation(self, x, y):
        return self.backend.library.multiply(x, y)


class Divide(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
        y: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x, y)

    def torch_implementation(self, x, y):
        return self.backend.library.true_divide(x, y)

    def tensorflow_implementation(self, x, y):
        return self.backend.library.truediv(x, y)


class Mod(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
        y: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x, y)

    def torch_implementation(self, x, y):
        return self.backend.library.remainder(x, y)

    def tensorflow_implementation(self, x, y):
        return self.backend.library.mod(x, y)


# endregion


# region: Powers and roots


class Square(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.square(x)


class Sqrt(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.sqrt(x)


class Power(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
        y: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.pow(x, y)


# endregion


# region: Exponential and logarithmic functions


class Exp(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.exp(x)


class Log(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.log(x)


class Log2(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.log2(x)

    def tensorflow_implementation(self, x):
        return self.backend.library.keras.ops.log2(x)


class Log10(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.log10(x)

    def tensorflow_implementation(self, x):
        return self.backend.library.math.log10(x)


# endregion


# region: Trigonometric functions


class Sin(BackendNode):
    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.sin(x)


class Cos(BackendNode):
    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.cos(x)


class Tan(BackendNode):
    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.tan(x)


class ArcSin(BackendNode):
    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.arcsin(x)

    def tensorflow_implementation(self, x):
        return self.backend.library.asin(x)


class ArcCos(BackendNode):
    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.arccos(x)

    def tensorflow_implementation(self, x):
        return self.backend.library.acos(x)


class ArcTan(BackendNode):
    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.arctan(x)

    def tensorflow_implementation(self, x):
        return self.backend.library.atan(x)


# endregion


# region: Other useful math functions


class Abs(BackendNode):
    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.abs(x)


class Floor(BackendNode):
    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.floor(x)


class Ceil(BackendNode):
    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.ceil(x)


class Maximum(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
        y: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.maximum(x, y)


class Minimum(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
        y: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.minimum(x, y)


# endregion


# region: Matrix operations


class MatMul(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
        y: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.matmul(x, y)


class SVD(BackendNode):
    def forward(self, x: Annotated[Any, DataConfiguration([])]) -> tuple[
        Annotated[Any, DataConfiguration([])],
        Annotated[Any, DataConfiguration([])],
        Annotated[Any, DataConfiguration([])],
    ]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.svd(x)

    def tensorflow_implementation(self, x):
        s, u, v = self.backend.library.linalg.svd(x)
        return u, s, v


# endregion


# region: Statistic operations


class Mean(BackendNode):
    """
    Computes the mean of the input tensor along the specified axis.
    """

    def __init__(
        self,
        axis: type[NO_DEFAULT] | None | int | tuple[int] = NO_DEFAULT,
        keepdims=False,
        backend=DEFAULT_DL_BACKEND,
    ):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(name=None, backend=backend)

    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x)

    def torch_implementation(self, x):
        if self.axis is NO_DEFAULT or self.axis is None:
            return self.backend.library.mean(x)
        return self.backend.library.mean(x, dim=self.axis, keepdim=self.keepdims)

    def tensorflow_implementation(self, x):
        if self.axis is NO_DEFAULT or self.axis is None:
            return self.backend.library.reduce_mean(x)
        return self.backend.library.reduce_mean(x, axis=self.axis, keepdims=self.keepdims)


class Std(BackendNode):
    def __init__(
        self,
        axis: type[NO_DEFAULT] | None | int | tuple[int] = NO_DEFAULT,
        keepdims=False,
        backend=DEFAULT_DL_BACKEND,
    ):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(name=None, backend=backend)

    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x)

    def torch_implementation(self, x):
        if self.axis is NO_DEFAULT or self.axis is None:
            return self.backend.library.std(x)
        return self.backend.library.std(x, dim=self.axis, keepdim=self.keepdims)

    def tensorflow_implementation(self, x):
        if self.axis is NO_DEFAULT or self.axis is None:
            return self.backend.library.reduce_std(x)
        return self.backend.library.reduce_std(x, axis=self.axis, keepdims=self.keepdims)


# endregion


# region: Reshaping operations


class Flatten(BackendNode):
    def __init__(
        self,
        start_dim: int = 0,
        end_dim: int = -1,
        backend=DEFAULT_DL_BACKEND,
    ):
        self.start_dim = start_dim
        self.end_dim = end_dim
        super().__init__(name=None, backend=backend)

    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.flatten(
            x, start_dim=self.start_dim, end_dim=self.end_dim
        )


class Transpose(BackendNode):
    def __init__(
        self,
        perm: list | None = None,
        backend=DEFAULT_DL_BACKEND,
    ):
        self.perm = perm if perm is not None else [1, 0]
        super().__init__(name=None, backend=backend)

    def forward(
        self, x: Annotated[Any, DataConfiguration([])]
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.permute(x, self.perm)

    def tensorflow_implementation(self, x):
        return self.backend.library.transpose(x, perm=self.perm)


# endregion


# endregion
