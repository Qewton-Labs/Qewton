from ..base import Node, InputPort, OutputPort
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
        self.in_port = InputPort(self.input_data_config, self)
        if (
            input_data_config.feature_axis is ...
            or input_data_config.feature_axis.variables is None
        ):
            raise ValueError("No variables found in this data configuration!")

        self.variables: Variable = output_variable
        assert (
            self.variables in input_data_config.feature_axis.variables
        ), f"{self.variables} are not inside the input data configuration."

        self.out_port = OutputPort(input_data_config[self.variables], self)

        self.axis_index = self.input_data_config.get_axis_indices_of_variables(
            self.variables
        )

    def run(self):
        data = self.in_port.value
        slices = [slice(None)] * len(self.input_data_config)
        slices[self.input_data_config.feature_axis_idx] = self.axis_index  # type: ignore
        self.out_port.set_value(data[tuple(slices)])


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
        self.in_port = InputPort(self.input_data_config, self)
        if (
            input_data_config.feature_axis is ...
            or input_data_config.feature_axis.variables is None
        ):
            raise ValueError("No variables found in this data configuration!")

        self.variables: Variable = input_data_config.feature_axis.variables
        # Save the output ports and the dimension we have to slice
        self._output_ports = []
        for key, dim in self.variables.items():
            self._output_ports.append(
                OutputPort(
                    input_data_config[Variable(key, dim)], self, name=f"OutputPort_{key}"
                )
            )

    def _run(self):
        data = self.in_port.value
        counter = 0
        start_dim = 0
        for dim in self.variables.values():
            slices = [slice(None)] * len(self.input_data_config)
            slices[self.input_data_config.feature_axis_idx] = slice(
                start_dim, start_dim + dim
            )
            self._output_ports[counter].set_value(data[tuple(slices)])  # type: ignore
            start_dim += dim
            counter += 1
