from typing import Annotated

from ...config.data_configurations import DataConfiguration
from ...config.axes import EllipsisAxes
from ..backend_node import BackendNode, TensorType


class ReLU(BackendNode[TensorType]):
    """General ReLU Class."""

    ellipsis_axes = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Annotated[TensorType, DataConfiguration(ellipsis_axes)]:
        return self.backend.library.relu(x)


class Tanh(BackendNode[TensorType]):
    ellipsis_axes = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Annotated[TensorType, DataConfiguration(ellipsis_axes)]:
        return self.backend.library.tanh(x)


class Sigmoid(BackendNode[TensorType]):
    """General Sigmoid Class."""

    ellipsis_axes = EllipsisAxes()

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Annotated[TensorType, DataConfiguration(ellipsis_axes)]:
        return self.backend.library.sigmoid(x)
