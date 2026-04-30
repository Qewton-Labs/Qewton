from typing import Annotated, Any

from ...config.configuration_base import DataConfiguration
from ..building_blocks.math import (
    Mean,
    Std,
)
from ..backend import DEFAULT_DL_BACKEND, Backend
from ...data.datasets.base import DataSet
from ...graphs.control_nodes.graph_node import TrackedNode


class StdNormalizationNode(TrackedNode):
    # TODO: Add more axis picking for more complex mean/std
    def __init__(
        self,
        dataset_node: DataSet,
        batch_axis: int = 0,
        divide_eps: float = 1.0e-5,
        backend: Backend = DEFAULT_DL_BACKEND,
        name="StdNormalization",
    ):
        self.batch_axis = batch_axis
        self.divide_eps = divide_eps

        mean_node = Mean(axis=self.batch_axis, keepdims=True, backend=backend)
        std_node = Std(axis=self.batch_axis, keepdims=True, backend=backend)
        self.mean = mean_node(dataset_node.data)
        self.std = std_node(dataset_node.data) + self.divide_eps
        super().__init__(name=name)

    def forward(
        self, x: Annotated[Any, DataConfiguration.empty()]
    ) -> Annotated[Any, DataConfiguration.empty()]:
        return (x - self.mean) / self.std


class InverseStdNormalization(StdNormalizationNode):

    def forward(
        self, x: Annotated[Any, DataConfiguration.empty()]
    ) -> Annotated[Any, DataConfiguration.empty()]:
        return x * self.std + self.mean
