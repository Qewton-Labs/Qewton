from copy import deepcopy
from types import EllipsisType
from typing import Annotated, Any

from qewton.backends import DEFAULT_DL_BACKEND, TensorType
from qewton.backends.base import DeepLearningBackend
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import EllipsisAxes, FeatureAxes, AxesDim
from qewton.config.variables import Variable
from qewton.graphs.nodes import NO_DEFAULT, NodeConfig, Port, InputPort, OutputPort, Node


# region: Slicing and value setting
class SetItem(Node[TensorType]):
    """Set item node that assigns values to specific indices of an input tensor.
    Has variable input ports for tensor and indices, and one output port.
    """

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
    """Slice node that extracts a portion of the input tensor based on slice configuration.
    Supports integer indices, slices, variables, and ellipsis notation.
    Has one input port and one output port.
    """

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

    @classmethod
    def _string_to_slice(cls, slice_str: str):
        # Convert a string representation of a slice to an actual slice object
        if slice_str == "...":
            return Ellipsis
        # Slices are saved as "slice(2, None, None)", so we split at commas
        # and also need to remove the "slice(" and ")" parts as well as any
        # whitespace
        elif slice_str.startswith("slice"):
            slice_str = slice_str.replace(" ", "")
            slice_parts = slice_str[6:-1].split(",")
            start = int(slice_parts[0]) if slice_parts[0] != "None" else None
            stop = int(slice_parts[1]) if slice_parts[1] != "None" else None
            step = int(slice_parts[2]) if slice_parts[2] != "None" else None
            return slice(start, stop, step)
        else:
            return int(slice_str)

    @classmethod
    def load_from_config(cls, config: NodeConfig) -> Node:
        if isinstance(config.other_args["slice_config"], str):
            config.other_args["slice_config"] = cls._string_to_slice(
                config.other_args["slice_config"]
            )
        elif isinstance(config.other_args["slice_config"], tuple):
            slice_tuple = tuple(
                cls._string_to_slice(s) for s in config.other_args["slice_config"]
            )
            config.other_args["slice_config"] = slice_tuple
        return super().load_from_config(config)


class SplitVariables(Node[TensorType]):
    """Split variables node that separates an input tensor into multiple outputs based on variable configuration.
    Has one input port and multiple output ports (one per variable).
    """

    def __init__(
        self,
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(
            name if name is not None else "SplitVariablesNode", backend=backend
        )
        self.split_dim = None
        self.backend: type[DeepLearningBackend[TensorType]] = backend
        self.split_sections = None

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
            self.split_dim = dynamic_configs[updated_port].feature_idx

        if isinstance(updated_port, InputPort) and len(self.output_ports) == 0:
            config_vars = dynamic_configs[updated_port].variables
            if isinstance(config_vars, Variable):
                copy_memo = {}
                in_config = deepcopy(dynamic_configs[updated_port], copy_memo)
                dynamic_configs[updated_port] = in_config
                self.split_sections = []
                for var_name, var_dim in config_vars.items():
                    self.split_sections.append(var_dim)
                    out_config = deepcopy(dynamic_configs[updated_port], copy_memo)
                    # since we iterate multiple times over the same config, we need
                    # to avoid reuse of the previous deepcopy
                    copy_memo.pop(id(dynamic_configs[updated_port]))
                    out_config.replace_feature_axes(
                        FeatureAxes(variable=Variable(var_name, var_dim))
                    )
                    new_out_port = OutputPort(out_config, self, name=var_name)
                    self.output_ports.append(new_out_port)
                    dynamic_configs[new_out_port] = out_config

        return updated_ports

    def get_output_port(self, name: str | Variable):
        # TODO Merge into general node?
        if isinstance(name, str):
            return super().get_output_port(name)
        if isinstance(name, Variable):
            for port in self.output_ports:
                if port.name == name.name:
                    return port
        raise ValueError(f"No output port with name {name} found in node {self.name}.")


class ConcatVariables(Node[TensorType]):
    """Concatenate variables node that combines multiple input tensors with different variables along a dimension.
    Assumes the feature axes are all the last axes.
    Has multiple input ports (one per variable) and one output port.
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
        out_var = Variable()
        for var in self.in_variables:
            out_var *= var
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
            for key in var.keys():
                if key in seen_keys:
                    raise ValueError(
                        f"Variable key '{key}' is not unique across input variables."
                    )
                seen_keys.add(key)

    def forward(self, *inp):
        return self.backend.math.concatenate(inp, axis=self.concat_dim)


class ConcatNode(Node[TensorType]):
    """Concatenate node that combines multiple input tensors along a specified dimension.
    Has multiple input ports and one output port.
    """

    def __init__(
        self, concat_dim: int, num_of_input_ports: int = 2, backend=DEFAULT_DL_BACKEND
    ):
        self.concat_dim = concat_dim
        self.num_of_input_ports = num_of_input_ports
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
    """Narrow node that returns a narrowed view of the input tensor along a specified dimension.
    Has one input port and one output port.
    """

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
    """Squeeze node that removes a single dimension of size 1 from the input tensor.
    Has one input port and one output port.
    """

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
    """Unsqueeze node that adds a new dimension of size 1 at the specified position.
    Has one input port and one output port.
    """

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
    """Reshape node that changes the shape of the input tensor to a specified new shape.
    Has one input port and one output port.
    """

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
    """Reshape at dimension node that reshapes a portion of the input tensor at a specified dimension.
    Has one input port and one output port.
    """

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
    """Flatten node that flattens the input tensor between specified dimensions.
    Has one input port and one output port.
    """

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
    """Unflatten node that reshapes a single dimension into multiple dimensions.
    Has one input port and one output port.
    """

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
    """Get shape node that returns the shape tuple of the input tensor.
    Has one input port and one output port.
    """

    def forward(
        self, x: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[tuple[int, ...], DataConfiguration.empty()]:
        return tuple(x.shape)


# endregion
