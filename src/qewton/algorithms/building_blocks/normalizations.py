from typing import Annotated, Any

from qewton.algorithms.building_blocks.math import Mean, Std
from qewton.algorithms.building_blocks.parameters import ParameterNode
from qewton.backends import DEFAULT_DL_BACKEND, Backend
from qewton.config.data_configurations import DataConfiguration
from qewton.data.datasets.base import DataSet
from qewton.graphs.control_nodes.graph_node import TrackedNode
from qewton.graphs.nodes import NodeState


class StdNormalizationNode(TrackedNode):
    # TODO: Add better axis picking/splitting for more complex mean/std
    def __init__(
        self,
        dataset_node: DataSet,
        normalization_axis: int | tuple[int] | None = 0,
        divide_eps: float = 1.0e-5,
        backend: Backend = DEFAULT_DL_BACKEND,
        name="StdNormalization",
    ):
        self.normalization_axis = normalization_axis
        self.divide_eps = divide_eps

        mean_node = Mean(axis=self.normalization_axis, keepdims=True, backend=backend)
        std_node = Std(axis=self.normalization_axis, keepdims=True, backend=backend)
        data_mean = mean_node(dataset_node.data)
        data_std = std_node(dataset_node.data) + self.divide_eps
        self.mean = ParameterNode(
            data_mean.shape, initial_value=data_mean, name="Data Mean " + name
        )
        self.std = ParameterNode(
            data_std.shape, initial_value=data_std, name="Data Std " + name
        )
        self.mean.setup()
        self.std.setup()
        self.mean.fix_node_state()
        self.std.fix_node_state()
        self._state = NodeState.FIXED
        super().__init__(name=name)

    def forward(
        self, x: Annotated[Any, DataConfiguration.empty()]
    ) -> Annotated[Any, DataConfiguration.empty()]:
        """Shifts the input by the mean of the data and scales by the inverse
        of the standard derivation.

        Parameters:
            x (ArrayLike): The data that should be normalized.

        Returns:
            (ArrayLike): The normalized data.
        """
        m = self.mean()
        std = self.std()
        return (x - m) / std

    def to(self, device):
        self.mean.to(device)
        self.std.to(device)


class InverseStdNormalization(StdNormalizationNode):

    def forward(
        self, x: Annotated[Any, DataConfiguration.empty()]
    ) -> Annotated[Any, DataConfiguration.empty()]:
        m = self.mean()
        std = self.std()
        return x * std + m
