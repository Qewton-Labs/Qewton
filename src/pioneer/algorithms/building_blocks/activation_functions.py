from typing import Annotated

from ...config.configuration_base import DataConfiguration
from ..backend_node import BackendNode, TensorType


class ReLU(BackendNode[TensorType]):
    """General ReLU Class."""

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration([])],
    ) -> Annotated[TensorType, DataConfiguration([])]:
        return self.backend.library.relu(x)


class Tanh(BackendNode[TensorType]):
    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration([])],
    ) -> Annotated[TensorType, DataConfiguration([])]:
        return self.backend.library.tanh(x)


class Sigmoid(BackendNode[TensorType]):
    """General Sigmoid Class."""

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration([])],
    ) -> Annotated[TensorType, DataConfiguration([])]:
        return self.backend.library.sigmoid(x)
