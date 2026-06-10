from copy import deepcopy
from types import EllipsisType
from typing import Annotated

from qewton.algorithms.backend_node import BackendNode
from qewton.backends import DEFAULT_DL_BACKEND, TensorType
from qewton.backends.base import DeepLearningBackend
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import EllipsisAxes, FeatureAxes
from qewton.config.variables import Variable
from qewton.graphs.nodes import NO_DEFAULT, Port, InputPort, OutputPort


class Narrow(BackendNode[TensorType]):
    def __init__(self, dim=None, start=0, length=None, backend=DEFAULT_DL_BACKEND):
        self.dim = dim if dim is not None else NO_DEFAULT
        self.start = start
        self.length = length if length is not None else NO_DEFAULT
        super().__init__(name=None, backend=backend)

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration([])],
    ) -> Annotated[TensorType, DataConfiguration([])]:
        return self.backend.math.narrow(x)


class Slice(BackendNode[TensorType]):

    def __init__(
        self,
        slice_config: (
            int | slice | Variable | tuple[slice | list[int] | EllipsisType | int, ...]
        ),
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.slice_obj = slice_config
        self.slice_config = slice_config
        super().__init__(name if name is not None else "SliceNode", backend=backend)

    def forward(
        self, inp: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        return self.implementation(inp)

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

    def torch_implementation(self, inp):
        return inp[self.slice_config]


class SplitVariables(BackendNode[TensorType]):

    def __init__(
        self,
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(
            name if name is not None else "SplitVariablesNode", backend=backend
        )
        self.split_dim = None
        self.split_sections = None

    def forward(self, inp: Annotated[TensorType, DataConfiguration.empty()]):
        return self.backend.math.split(inp, self.split_sections, dim=self.split_dim)

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


class ConcatVariables(BackendNode[TensorType]):
    """
    Assumes the feature axes are all the last axes.
    """

    def __init__(
        self,
        in_variables,
        name=None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(
            name if name is not None else "ConcatVariablesNode", backend=backend
        )

        self.in_variables = in_variables
        self.check_unique_var_keys()
        self.concat_dim = -1
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
