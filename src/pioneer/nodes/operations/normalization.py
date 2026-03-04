from typing import Any

from ..base import Node, Port
from ...data.datasets.base import DataSet
from ...config.configuration_base import DataConfiguration
from ...optim.hyperparameter.base import BooleanHyperparameter, HyperParameter


class NormalizationNode(Node):
    """Normalizes the input by shifting it by the mean value and dividing by the standard
    derivation.
    """

    def __init__(
        self,
        data_config: DataConfiguration,
        dataset_node: DataSet,
        active: bool | BooleanHyperparameter = True,
        name: str = "Normalization",
    ) -> None:
        """
        Args:
            data_config (DataConfiguration): The expected data shape.
            dataset_node (DataSet): The data set where the data originates from.
            name (str, optional): The name of this node. Defaults to "NormalizationNode".

        Raises:
            ValueError: If the provided DataConfiguration does not contain
                any variables, we can use for normalization or when the dataset does not
                fit the expected data.
        """
        super().__init__(name=name)
        self.data_config = data_config
        self.dataset_node = dataset_node
        self.is_active = HyperParameter.from_value(active, "Normalization Active")
        self.port = Port(self.data_config, self, "port", True)
        if data_config.feature_axis is ... or data_config.feature_axis.variables is None:
            raise ValueError("No variables found in this data configuration!")
        if (
            self.dataset_node.data_config.feature_axis is ...
            or self.dataset_node.data_config.feature_axis.variables is None
        ):
            raise ValueError("No variables found in the data set!")
        if (
            not data_config.feature_axis.variables
            in self.dataset_node.data_config.feature_axis.variables
        ):
            raise ValueError(
                f"Can not normalize data not inside the dataset {dataset_node}!"
            )

    @property
    def input_ports(self) -> dict[str, Port]:
        return {self.InputKeys.INPUT: self.port}

    @property
    def output_ports(self) -> dict[str, Port]:
        return {self.OutputKeys.OUTPUT: self.port}

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        data = inputs[self.InputKeys.INPUT]
        if self.is_active.value:
            data_mean, data_std = self._extract_mean_and_std()
            return {self.OutputKeys.OUTPUT: (data - data_mean) / data_std}
        return {self.OutputKeys.OUTPUT: data}

    def _extract_mean_and_std(self):
        slices = [slice(None)] * len(self.data_config)
        data_indices = self.dataset_node.data_config.get_axis_indices_of_variables(
            self.data_config.feature_axis.variables  # type: ignore
        )
        slices[self.data_config.feature_axis_idx] = data_indices  # type: ignore
        slices = tuple(slices)
        data_mean = self.dataset_node.mean[slices]
        data_std = self.dataset_node.std[slices] + self.dataset_node.std_eps
        return data_mean, data_std

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        """Returns tunable parameters of this node."""
        return [self.is_active]


class InverseNormalizationNode(NormalizationNode):
    """Does an inverse normalization of the input by multiplying by the standard
    derivation and adding the mean value given in the dataset.
    """

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if inputs is None:
            raise ValueError("Input can not be None!")
        data = inputs[self.InputKeys.INPUT]
        if self.is_active.value:
            data_mean, data_std = self._extract_mean_and_std()
            return {self.OutputKeys.OUTPUT: data * data_std + data_mean}
        return {self.OutputKeys.OUTPUT: data}
