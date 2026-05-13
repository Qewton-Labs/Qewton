from typing import Annotated

from ..backend import DEFAULT_DL_BACKEND, TensorType
from ..backend_node import BackendNode

from ...config.data_configurations import DataConfiguration as DC
from ...config.axes import EllipsisAxes, AxesDim, FeatureAxes, MinimumDim
from ...graphs.nodes import NO_DEFAULT, InputPort, Port

# The following classes represent basic mathematical operations.
# They are designed to work with different operations and one only needs to pass
# in the name of the respective function in the backend to create a new operation.
# This reduces the number of classes we need to create. However, if there are
# specific operations that require special handling, we can of course
# create separate classes for them.


# region: Arithmetic operations


class Add(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.add(x, y)


class Subtract(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.implementation(x, y)

    def torch_implementation(self, x, y):
        return self.backend.library.sub(x, y)

    def tensorflow_implementation(self, x, y):
        return self.backend.library.subtract(x, y)


class Multiply(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.implementation(x, y)

    def torch_implementation(self, x, y):
        return self.backend.library.mul(x, y)

    def tensorflow_implementation(self, x, y):
        return self.backend.library.multiply(x, y)


class Divide(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.implementation(x, y)

    def torch_implementation(self, x, y):
        return self.backend.library.true_divide(x, y)

    def tensorflow_implementation(self, x, y):
        return self.backend.library.truediv(x, y)


class Mod(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.implementation(x, y)

    def torch_implementation(self, x, y):
        return self.backend.library.remainder(x, y)

    def tensorflow_implementation(self, x, y):
        return self.backend.library.mod(x, y)


# endregion


# region: Powers and roots


class Square(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.square(x)


class Sqrt(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.sqrt(x)


class Power(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.pow(x, y)


# endregion


# region: Exponential and logarithmic functions


class Exp(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.exp(x)


class Log(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.log(x)


class Log2(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.log2(x)

    def tensorflow_implementation(self, x):
        return self.backend.library.keras.ops.log2(x)


class Log10(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.log10(x)

    def tensorflow_implementation(self, x):
        return self.backend.library.math.log10(x)


# endregion


# region: Trigonometric functions


class Sin(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.sin(x)


class Cos(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.cos(x)


class Tan(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.tan(x)


class ArcSin(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.arcsin(x)

    def tensorflow_implementation(self, x):
        return self.backend.library.asin(x)


class ArcCos(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.arccos(x)

    def tensorflow_implementation(self, x):
        return self.backend.library.acos(x)


class ArcTan(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.arctan(x)

    def tensorflow_implementation(self, x):
        return self.backend.library.atan(x)


# endregion


# region: Other useful math functions


class Abs(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.abs(x)


class Floor(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.floor(x)


class Ceil(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.ceil(x)


class Maximum(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.maximum(x, y)


class Minimum(BackendNode[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.library.minimum(x, y)


# endregion


# region: Matrix operations


class MatMul(BackendNode[TensorType]):
    ell_ax = EllipsisAxes()
    dim_1 = AxesDim(None)
    dim_2 = AxesDim(None)

    def forward(
        self,
        x: Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_1,)))],
        y: Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_1, dim_2)))],
    ) -> Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_2,)))]:
        return self.backend.library.matmul(x, y)


class SVD(BackendNode[TensorType]):
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
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.svd(x)

    def tensorflow_implementation(self, x):
        s, u, v = self.backend.library.linalg.svd(x)
        return u, s, v


# endregion


# region: Statistic operations


class Mean(BackendNode[TensorType]):
    """
    Computes the mean of the input tensor along the specified axis.
    """

    def __init__(
        self,
        axis: None | int | tuple[int] = None,
        keepdims=False,
        backend=DEFAULT_DL_BACKEND,
    ):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(name=None, backend=backend)

    def update_data_configs(
        self, updated_port, config_dict, dynamic_configs: dict[Port, DC]
    ):
        port_config = dynamic_configs[updated_port]
        port_config_was_updated = port_config.update_config(config_dict)
        changed_ports = set()
        # TODO: Finish this
        # if port_config_was_updated:
        #     # if the port changed we also update the other one
        #     if isinstance(updated_port, InputPort):
        #         out_config = dynamic_configs[self.output_ports[0]]
        #         if not self.keepdims:
        #             if self.axis is None:
        #                 out_config.axes = (FeatureAxes(shape=(AxesDim(1),)),)
        #             elif isinstance(self.axis, int):
        #                 pass
        #         else:
        #             pass

        #     else:
        #         changed_ports.add(updated_port)
        #         in_port = self.input_ports[0]
        #         if self.axis is None:
        #             pass
        #         out_port = updated_port

        return changed_ports

    def forward(
        self, x: Annotated[TensorType, DC(EllipsisAxes())]
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.mean(x, dim=self.axis, keepdim=self.keepdims)

    def tensorflow_implementation(self, x):
        return self.backend.library.reduce_mean(x, axis=self.axis, keepdims=self.keepdims)


class Std(BackendNode[TensorType]):
    def __init__(
        self,
        axis: type[NO_DEFAULT] | None | int | tuple[int] = NO_DEFAULT,
        keepdims=False,
        backend=DEFAULT_DL_BACKEND,
    ):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(name=None, backend=backend)

    def forward(self, x: Annotated[TensorType, DC([])]) -> Annotated[TensorType, DC([])]:
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


class Flatten(BackendNode[TensorType]):
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
        self, x: Annotated[TensorType, DC.empty()]
    ) -> Annotated[TensorType, DC.empty()]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.flatten(
            x, start_dim=self.start_dim, end_dim=self.end_dim
        )


class Transpose(BackendNode[TensorType]):
    def __init__(
        self,
        perm: list | None = None,
        backend=DEFAULT_DL_BACKEND,
    ):
        self.perm = perm if perm is not None else [1, 0]
        super().__init__(name=None, backend=backend)

    def x_data_config(self):
        return DC.empty()

    def forward(
        self, x: Annotated[TensorType, x_data_config]
    ) -> Annotated[TensorType, DC.empty()]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.permute(x, self.perm)

    def tensorflow_implementation(self, x):
        return self.backend.library.transpose(x, perm=self.perm)


# endregion


# endregion
