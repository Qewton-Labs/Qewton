from typing import Any

from ..base import Node, Port
from ...config.variables import Variable
from ...config.configuration_base import DataConfiguration


class SliceNode(Node):
    """Extracts the variables of a dataset by slicing.

    TODO: Just assumes that slicing works
    """

    def __init__(
        self,
        input_data_config: DataConfiguration,
        output_variable: Variable,
        name: str = "Slice",
    ) -> None:
        """
        Args:
            input_data_config (DataConfiguration): The expected data shape.
            output_variable (Variable): The variables that should be sliced from the data.
            name (str, optional): The name of this node. Defaults to "Slice".

        Raises:
            ValueError: If the provided DataConfiguration does not contain
                any variables, we can not slice along the variables.
        """
        super().__init__(name=name)
        self.input_data_config = input_data_config
        self._in_port = Port(self.input_data_config, self, "port", True)
        if (
            input_data_config.feature_axis is ...
            or input_data_config.feature_axis.variables is None
        ):
            raise ValueError("No variables found in this data configuration!")

        self.variables: Variable = output_variable
        assert (
            self.variables in input_data_config.feature_axis.variables
        ), f"{self.variables} are not inside the input data configuration."

        self._out_port = Port(input_data_config[self.variables], self, "port", True)

        self.axis_index = self.input_data_config.get_axis_indices_of_variables(
            self.variables
        )

    @property
    def input_ports(self) -> dict[str, Port]:
        return {self.InputKeys.INPUT: self._in_port}

    @property
    def output_ports(self) -> dict[str, Port]:
        return {self.OutputKeys.OUTPUT: self._out_port}

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        data = inputs[self.InputKeys.INPUT]
        slices = [slice(None)] * len(self.input_data_config)
        slices[self.input_data_config.feature_axis_idx] = self.axis_index  # type: ignore
        sliced_data = data[tuple(slices)]
        return {self.OutputKeys.OUTPUT: sliced_data}


class SplitNode(Node):
    """Splits the input into all variables.

    TODO: Just assumes that slicing works to split the data
    """

    def __init__(self, input_data_config: DataConfiguration, name: str = "Split") -> None:
        """
        Args:
            input_data_config (DataConfiguration): The expected data shape.
            name (str, optional): The name of this node. Defaults to "Split".

        Raises:
            ValueError: If the provided DataConfiguration does not contain
                any variables, we can not slice along the variables.
        """
        super().__init__(name=name)
        self.input_data_config = input_data_config
        self._in_port = Port(self.input_data_config, self, "port", True)
        if (
            input_data_config.feature_axis is ...
            or input_data_config.feature_axis.variables is None
        ):
            raise ValueError("No variables found in this data configuration!")

        self.variables: Variable = input_data_config.feature_axis.variables
        # Save the output ports and the dimension we have to slice
        self._out_ports: dict[str, Port] = {}
        for key, dim in self.variables.items():
            self._out_ports[key] = Port(input_data_config[Variable(key, dim)], self, key)

    @property
    def input_ports(self) -> dict[str, Port]:
        return {self.InputKeys.INPUT: self._in_port}

    @property
    def output_ports(self) -> dict[str, Port]:
        return self._out_ports

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        data = inputs[self.InputKeys.INPUT]
        output_dict = {}
        start_dim = 0
        for key, dim in self.variables.items():
            slices = [slice(None)] * len(self.input_data_config)
            slices[self.input_data_config.feature_axis_idx] = slice(
                start_dim, start_dim + dim
            )
            output_dict[key] = data[tuple(slices)]
            start_dim += dim
        return output_dict
