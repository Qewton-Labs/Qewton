from copy import deepcopy
from typing import Annotated

from qewton.backends import DEFAULT_DL_BACKEND, TensorType, Backend, DeepLearningBackend
from qewton.graphs.nodes import Node
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.config.errors import DataConfigMismatchError
from qewton.config.axes import (
    EllipsisAxes,
    AxesDim,
    EllipsisDim,
    FeatureAxes,
    MinimumDim,
)
from qewton.graphs.nodes import InputPort, Port

# The following classes represent basic mathematical operations.
# They are designed to work with different operations and one only needs to pass
# in the name of the respective function in the backend to create a new operation.
# This reduces the number of classes we need to create. However, if there are
# specific operations that require special handling, we can of course
# create separate classes for them.


# region: Arithmetic operations


class Add(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.add(x, y)


class Subtract(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.subtract(x, y)


class Multiply(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.multiply(x, y)


class Divide(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        # TODO: should this rather be true divide?
        return self.backend.math.divide(x, y)


class Mod(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.mod(x, y)


# endregion


# region: Powers and roots


class Square(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.square(x)


class Sqrt(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.sqrt(x)


class Power(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def __init__(
        self,
        power: float | None = None,
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(name, backend)
        if power is not None:
            self.input_ports[-1].default = power

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.power(x, y)


# endregion


# region: Exponential and logarithmic functions


class Exp(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.exp(x)


class Log(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.log(x)


class Log2(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.log2(x)


class Log10(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.log10(x)


# endregion


# region: Trigonometric functions


class Sin(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.sin(x)


class Cos(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.cos(x)


class Tan(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.tan(x)


class ArcSin(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.arcsin(x)


class ArcCos(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.arccos(x)


class ArcTan(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.arctan(x)


# endregion


# region: Other useful math functions


class Abs(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.abs(x)


class Floor(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.floor(x)


class Ceil(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.ceil(x)


class Maximum(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.maximum(x, y)


class Minimum(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DC(ellipsis_dims)],
        y: Annotated[TensorType, DC(ellipsis_dims)],
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.minimum(x, y)


# endregion


# region: Matrix operations


class MatMul(Node[TensorType]):
    ell_ax = EllipsisAxes()
    dim_1 = AxesDim(None)
    dim_2 = AxesDim(None)

    def forward(
        self,
        x: Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_1,)))],
        y: Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_1, dim_2)))],
    ) -> Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_2,)))]:
        return self.backend.math.matmul(x, y)


class SVD(Node[TensorType]):
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


# endregion


# region: Statistic operations
class ReductionNode(Node[TensorType]):

    def __init__(
        self,
        name=None,
        axis: None | int | tuple[int, ...] = None,
        keepdims=False,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(name=name, backend=backend)

    def update_data_configs(
        self, updated_port, config_dict, dynamic_configs: dict[Port, DC]
    ):
        port_config = dynamic_configs[updated_port]
        port_config_was_updated = port_config.update_config(config_dict)
        changed_ports = set()
        if port_config_was_updated:
            if isinstance(updated_port, InputPort):
                if not self.keepdims:
                    new_config = self._build_reduced_out_config(port_config)
                else:
                    new_config = self._build_keepdims_config(port_config, 1)
                if new_config is None:
                    # We can not construct the output config:
                    return changed_ports
                # Now check if the config is changed:
                out_config = dynamic_configs[self.output_ports[0]]  # old one
                unify_config = out_config.unify_with(new_config)
                if out_config.update_config(unify_config[0]):
                    changed_ports.add(self.output_ports[0])
            else:  # Output port was updated
                changed_ports.add(updated_port)
                if not self.keepdims:
                    # TODO: Here it is somewhat difficult to determine
                    # the expected input axis, since we remove them
                    # -> we would need some placeholder that acts
                    # over multiple axes/dims?
                    # Else we will check compatibility once we connect
                    # the input port.
                    pass
                else:  # keepdim case
                    # check if all dimensions are 1 in the output:
                    self._check_output_config_compatible(port_config)
                    # Reconstruct input
                    new_config = self._build_keepdims_config(port_config, None)
                    if new_config is not None:
                        old_in_config = dynamic_configs[self.input_ports[0]]
                        unify_config = old_in_config.unify_with(new_config)
                        if old_in_config.update_config(unify_config[0]):
                            changed_ports.add(self.input_ports[0])

        return changed_ports

    def _check_output_config_compatible(self, out_config):
        if self.axis is not None:
            return
        for axes in out_config.axes:
            for dim in axes.shape:
                if not isinstance(dim, EllipsisDim):
                    if dim.size != 1:
                        raise DataConfigMismatchError(
                            f"Output of mean needs to have size 1, but found {dim.size}"
                        )

    def _build_reduced_out_config(self, port_config: DC):
        """If we reduce the dimension (keepdims=False) we have to
        remove the corresponding dimensions"""
        new_config = deepcopy(port_config)
        mean_axes = (self.axis,) if isinstance(self.axis, int) else self.axis

        if mean_axes is None:
            # TODO: Is this the correct Axes to use?
            return DC(FeatureAxes(shape=(AxesDim(1),)))

        remove_elements = []
        for idx in mean_axes:
            axis_type, dim = new_config.get_axes_and_dim(idx)
            if axis_type is None or dim is None:
                return None
            remove_elements.append((axis_type, dim))

        for axis_type, dim in remove_elements:
            new_config.remove_dim(axis_type, dim)

        return new_config

    def _build_keepdims_config(self, port_config: DC, update_value: int | None):
        """If we keep the dimension (keepdims=True) we just have to
        set the corresponding dimensions to 1."""
        new_config = deepcopy(port_config)
        mean_axes = (self.axis,) if isinstance(self.axis, int) else self.axis
        max_value = max(mean_axes) if mean_axes is not None else None

        reduce_idx = 0
        replaced_all_dims = False
        for axis in new_config.axes:
            if isinstance(axis, EllipsisAxes):
                if mean_axes is None:
                    continue
                # We can not construct the config, because we can
                # not count the index
                return None

            for dim in axis.shape:
                if isinstance(dim, EllipsisDim):
                    if mean_axes is None:
                        continue
                    # We can not construct the config, because we can
                    # not count the index
                    return None
                if mean_axes is None or reduce_idx in mean_axes:
                    dim.update_size(update_value)

                reduce_idx += 1
                if max_value is not None and reduce_idx > max_value:
                    replaced_all_dims = True
                    break

            if replaced_all_dims:
                break

        return new_config


class Mean(ReductionNode[TensorType]):
    """
    Computes the mean of the input tensor along the specified axis.
    """

    def forward(
        self, x: Annotated[TensorType, DC(EllipsisAxes())]
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.mean(x, axis=self.axis, keepdims=self.keepdims)


class Sum(ReductionNode[TensorType]):
    """
    Computes the sum of the input tensor along the specified axis.
    """

    def forward(
        self, x: Annotated[TensorType, DC(EllipsisAxes())]
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.sum(x, axis=self.axis, keepdims=self.keepdims)


class Std(ReductionNode[TensorType]):

    def forward(
        self, x: Annotated[TensorType, DC(EllipsisAxes())]
    ) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.std(x, axis=self.axis, keepdims=self.keepdims)


# endregion


# region: Reshaping operations


class Flatten(Node[TensorType]):
    def __init__(
        self,
        start_dim: int = 0,
        end_dim: int = -1,
        backend=DEFAULT_DL_BACKEND,
    ):
        self.start_dim = start_dim
        self.end_dim = end_dim
        super().__init__(name=None, backend=backend)

    # TODO: Make config connection
    def forward(
        self, x: Annotated[TensorType, DC.empty()]
    ) -> Annotated[TensorType, DC.empty()]:
        return self.backend.math.flatten(
            x, start_dim=self.start_dim, end_dim=self.end_dim
        )


class Transpose(Node[TensorType]):
    def __init__(
        self,
        perm: list | None = None,
        backend=DEFAULT_DL_BACKEND,
    ):
        self.perm = perm if perm is not None else [1, 0]
        super().__init__(name=None, backend=backend)

    # TODO: Make config connection
    def x_data_config(self):
        return DC.empty()

    def forward(
        self, x: Annotated[TensorType, x_data_config]
    ) -> Annotated[TensorType, DC.empty()]:
        return self.backend.math.transpose(x, axes=self.perm)


# endregion


# endregion
