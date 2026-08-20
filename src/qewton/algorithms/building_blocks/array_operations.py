from copy import deepcopy
from types import EllipsisType
from typing import Annotated, Any

from qewton.backends import DEFAULT_DL_BACKEND, TensorType
from qewton.backends.base import DeepLearningBackend
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import EllipsisAxes, FeatureAxes, AxesDim
from qewton.config.variables import Variable
from qewton.graphs.nodes import NO_DEFAULT, Port, InputPort, OutputPort, Node


# region: Slicing and value setting
class SetItem(Node[TensorType]):
    data_axis = EllipsisAxes()

    def forward(
        self,
        inp: Annotated[TensorType, DataConfiguration(data_axis)],
        key: Annotated[Any, DataConfiguration.empty()],
        value: Annotated[TensorType, DataConfiguration.empty()],
    ) -> Annotated[TensorType, DataConfiguration(data_axis)]:
        inp[key] = value
        return inp

    def _track(self, *args, **kwargs):
        output_trackers = super()._track(*args, **kwargs)
        # Since set item happens in place, we have to update the
        # TrackingObject in place as well. Else while tracking this
        # operation happens at an arbitrary point.
        if "inp" in kwargs:
            inp_tacker = kwargs["inp"]
        else:
            inp_tacker = args[0]
        inp_tacker.to_ports = []
        inp_tacker.last_output_port = self.output_ports[0]
        return output_trackers


class Slice(Node[TensorType]):

    def __init__(
        self,
        slice_config: (
            int | slice | Variable | tuple[slice | list[int] | EllipsisType | int, ...]
        ),
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.slice_obj = slice_config
        self.backend: type[DeepLearningBackend[TensorType]] = backend
        self.slice_config = slice_config
        super().__init__(name if name is not None else "SliceNode", backend=backend)

    def forward(
        self, inp: Annotated[Any, DataConfiguration.empty()]
    ) -> Annotated[Any, DataConfiguration.empty()]:
        if isinstance(inp, self.backend.default_dtype):
            return self.backend.math.slice(inp, self.slice_config)
        return inp[self.slice_config]

    def update_data_configs(
        self, updated_port, config_dict, dynamic_configs: dict[Port, DataConfiguration]
    ):
        updated_ports = super().update_data_configs(
            updated_port, config_dict, dynamic_configs
        )
        if isinstance(self.slice_obj, Variable) and isinstance(updated_port, InputPort):
            try:
                self.slice_config = dynamic_configs[updated_port].get_variable_slice(
                    self.slice_obj
                )
                # TODO: Build new output config
            except AssertionError:
                # Could raise an assertion error if config is not concrete enough yet
                pass
        return updated_ports


class SplitVariables(Node[TensorType]):
    """Splits an input tensor's feature axis into one output port per
    Variable in `split_into`.

    The caller decides the exact pieces upfront - this used to auto-derive
    them from whatever named entries its own input happened to have, which
    forced always splitting down to the finest granularity available before
    anything downstream could ask for less. A caller that knows every
    downstream consumer in advance (e.g. PINNPipeline) can instead split
    only as finely as actually needed - a variable nothing ever asks to
    subdivide can be named whole in `split_into` and passed through with a
    single piece.

    Args:
        split_into (list[Variable]): The pieces to split the input's
            feature axis into, in order. Each must be a contiguous run of
            the eventual input's own feature Variable - not necessarily a
            single leaf, e.g. an auto-expanded 3D variable can be its own
            whole piece.
    """

    def __init__(
        self,
        split_into: list[Variable],
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(
            name if name is not None else "SplitVariablesNode", backend=backend
        )
        self.backend: type[DeepLearningBackend[TensorType]] = backend
        self.split_into = split_into
        self.split_dim = None
        self.split_sections = [var.dim for var in split_into]
        # A real FeatureAxes here, not a DataConfiguration.empty() placeholder
        # later overridden only in dynamic_configs: callers that inspect a
        # port's *static* config before anything is connected (e.g.
        # PINNPipeline's gradient-tracking pass) need `variables` to already
        # be correct - only the other axes (batch shape etc.), unknown until
        # the input actually connects, stay a wildcard here.
        ellipsis_axes = EllipsisAxes()
        self._output_ports = [
            OutputPort(
                DataConfiguration(
                    ellipsis_axes, FeatureAxes(variable=var), dtype=backend.default_dtype
                ),
                node=self,
                name=var.name,
            )
            for var in split_into
        ]

    def forward(self, inp: Annotated[TensorType, DataConfiguration.empty()]):
        return self.backend.math.split(
            inp, self.split_sections, axis=self.split_dim  # type: ignore
        )

    def update_data_configs(
        self, updated_port, config_dict, dynamic_configs: dict[Port, DataConfiguration]
    ):
        updated_ports = super().update_data_configs(
            updated_port, config_dict, dynamic_configs
        )
        if isinstance(updated_port, InputPort):
            in_config = dynamic_configs[updated_port]
            self.split_dim = in_config.feature_idx
            copy_memo = {}
            for out_port, var in zip(self.output_ports, self.split_into):
                out_config = deepcopy(in_config, copy_memo)
                # since we iterate multiple times over the same config, we need
                # to avoid reuse of the previous deepcopy
                copy_memo.pop(id(in_config), None)
                out_config.replace_feature_axes(FeatureAxes(variable=var))
                dynamic_configs[out_port] = out_config

        return updated_ports

    def get_output_port(self, name: str | Variable):
        # TODO Merge into general node?
        if isinstance(name, Variable):
            name = name.name
        return super().get_output_port(name)


class ConcatVariables(Node[TensorType]):
    """
    Assumes the feature axes are all the last axes.
    """

    def __init__(
        self,
        in_variables,
        concat_dim: int = -1,
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(
            name if name is not None else "ConcatVariablesNode", backend=backend
        )
        self.backend: type[DeepLearningBackend[TensorType]] = backend

        self.in_variables = in_variables
        self.check_unique_var_keys()
        self.concat_dim = concat_dim
        self.concat_sections = None

        ellipsis_axes = EllipsisAxes()

        self._input_ports = []
        for var in self.in_variables:
            self._input_ports.append(
                InputPort(
                    DataConfiguration(
                        ellipsis_axes,
                        FeatureAxes(variable=var),
                        dtype=self.backend.default_dtype,
                    ),
                    node=self,
                    name=var.name,
                )
            )
        # Fold starting from the first real variable, not an empty
        # Variable() placeholder - composing with a genuinely empty
        # Variable leaves its dim as None (Variable.__init__ propagates
        # "any child's dim is None" outward), which breaks arithmetic like
        # Variable.get_slice's running_idx += child.dim on the result.
        out_var = self.in_variables[0]
        for var in self.in_variables[1:]:
            out_var = out_var * var
        self._output_ports = [
            OutputPort(
                DataConfiguration(
                    ellipsis_axes,
                    FeatureAxes(variable=out_var),
                    dtype=self.backend.default_dtype,
                ),
                node=self,
                name=out_var.name,
            )
        ]

    def check_unique_var_keys(self):
        seen_keys = set()
        for var in self.in_variables:
            if var.name in seen_keys:
                raise ValueError(
                    f"Variable key '{var.name}' is not unique across input variables."
                )
            seen_keys.add(var.name)

    def forward(self, *inp):
        return self.backend.math.concatenate(inp, axis=self.concat_dim)


class ConcatNode(Node[TensorType]):
    def __init__(
        self, concat_dim: int, num_of_input_ports: int = 2, backend=DEFAULT_DL_BACKEND
    ):
        self.concat_dim = concat_dim
        super().__init__(name=None, backend=backend)
        self.backend: type[DeepLearningBackend[TensorType]] = backend

        self._input_ports = []
        for i in range(num_of_input_ports):
            self._input_ports.append(
                InputPort(
                    DataConfiguration.empty(),
                    node=self,
                    name=f"Input_{i}",
                )
            )

    def forward(self, *inp) -> Annotated[TensorType, DataConfiguration.empty()]:
        return self.backend.math.concatenate(inp, axis=self.concat_dim)


# endregion
# region: Reshaping


class Narrow(Node[TensorType]):
    def __init__(self, dim=None, start=0, length=None, backend=DEFAULT_DL_BACKEND):
        self.dim = dim if dim is not None else NO_DEFAULT
        self.start = start
        self.l = length if length is not None else NO_DEFAULT
        super().__init__(name=None, backend=backend)
        self.backend: type[DeepLearningBackend[TensorType]] = backend

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration.empty()],
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        o = self.backend.math.narrow(x, self.dim, self.start, self.l)  # type: ignore
        return o


class Squeeze(Node[TensorType]):

    def __init__(
        self,
        dim: int,
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(name if name is not None else "SqueezeNode", backend=backend)
        self.dim = dim
        self.backend: type[DeepLearningBackend[TensorType]] = backend

    def forward(
        self, inp: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        return self.backend.math.squeeze(inp, self.dim)

    def update_data_configs(
        self, updated_port, config_dict, dynamic_configs: dict[Port, DataConfiguration]
    ):
        updated_ports = super().update_data_configs(
            updated_port, config_dict, dynamic_configs
        )
        if isinstance(updated_port, InputPort):
            axes, index_dim = dynamic_configs[updated_port].get_axes_and_dim(self.dim)
            if axes is not None and index_dim is not None:
                new_output_config = deepcopy(dynamic_configs[updated_port])
                new_output_config.remove_dim(axes, index_dim)
                old_output_config = dynamic_configs[self.output_ports[0]]
                unify_config = old_output_config.unify_with(new_output_config)[0]
                output_changed = old_output_config.update_config(unify_config)
                if output_changed:
                    updated_ports.add(self.output_ports[0])
        return updated_ports


class Unsqueeze(Node[TensorType]):

    def __init__(
        self,
        dim: int,
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(name if name is not None else "UnsqueezeNode", backend=backend)
        self.dim = dim
        self.backend: type[DeepLearningBackend[TensorType]] = backend

    def forward(
        self, inp: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        return self.backend.math.unsqueeze(inp, self.dim)

    def update_data_configs(
        self, updated_port, config_dict, dynamic_configs: dict[Port, DataConfiguration]
    ):
        updated_ports = super().update_data_configs(
            updated_port, config_dict, dynamic_configs
        )
        if isinstance(updated_port, InputPort):
            axes, index_dim = dynamic_configs[updated_port].get_axes_and_dim(self.dim)
            if axes is not None and index_dim is not None:
                # Build new config and add a dimension
                new_output_config = deepcopy(dynamic_configs[updated_port])
                new_axes, new_dim = new_output_config.get_axes_and_dim(self.dim)
                dim_idx = new_axes.get_dim_idx(new_dim)  # type: ignore
                new_axes.add_dim(AxesDim(1), dim_idx + 1)  # type: ignore
                # Check if the old config is the same anyway
                old_output_config = dynamic_configs[self.output_ports[0]]
                unify_config = old_output_config.unify_with(new_output_config)[0]
                output_changed = old_output_config.update_config(unify_config)
                # If something changed we have to pass this through the graph
                if output_changed:
                    updated_ports.add(self.output_ports[0])
        return updated_ports


class Reshape(Node[TensorType]):

    def __init__(
        self,
        new_shape: tuple[int, ...],
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.new_shape = new_shape
        super().__init__(name if name is not None else "ReshapeNode", backend=backend)
        self.backend: type[DeepLearningBackend[TensorType]] = backend

    def forward(
        self, inp: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        return self.backend.math.reshape(inp, self.new_shape)


class ReshapeAtDim(Node[TensorType]):

    def __init__(
        self,
        dim: int,
        new_shape: tuple[int, ...],
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.new_shape = new_shape
        self.dim = dim

        super().__init__(name if name is not None else "ReshapeNode", backend=backend)
        self.backend: type[DeepLearningBackend[TensorType]] = backend

    def forward(
        self, inp: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        if self.dim < 0:
            self.dim = len(inp.shape) + self.dim
        new_shape = inp.shape[: self.dim] + self.new_shape + inp.shape[self.dim + 1 :]
        return self.backend.math.reshape(inp, new_shape)


class Flatten(Node[TensorType]):

    def __init__(
        self,
        start_dim: int = 0,
        end_dim: int = -1,
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.start_dim = start_dim
        self.end_dim = end_dim
        super().__init__(name if name is not None else "FlattenNode", backend=backend)
        self.backend: type[DeepLearningBackend[TensorType]] = backend

    def forward(
        self, inp: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        return self.backend.math.flatten(inp, self.start_dim, self.end_dim)


class Unflatten(Node[TensorType]):

    def __init__(
        self,
        axis: int,
        sizes: tuple[int, ...],
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.axis = axis
        self.sizes = sizes
        super().__init__(name if name is not None else "UnflattenNode", backend=backend)
        self.backend: type[DeepLearningBackend[TensorType]] = backend

    def forward(
        self, inp: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        return self.backend.math.unflatten(inp, self.axis, self.sizes)


# endregion

# region: Array info


class GetShapeNode(Node[TensorType]):

    def forward(
        self, x: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[tuple[int, ...], DataConfiguration.empty()]:
        return tuple(x.shape)


# endregion
