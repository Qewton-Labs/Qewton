from typing import Annotated, Any

from ...config.configuration_base import DataConfiguration
from ..backend_node import BackendNode


class ReLU(BackendNode):
    """General ReLU Class."""

    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.relu(x)


class Tanh(BackendNode):
    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.tanh(x)


class Sigmoid(BackendNode):
    """General Sigmoid Class."""

    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.sigmoid(x)
