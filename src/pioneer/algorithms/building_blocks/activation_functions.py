from typing import Annotated, Any

from ...config.configuration_base import DataConfiguration
from ..base import BackendNode


class ReLU(BackendNode):
    """General ReLU Class."""

    def __call__(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.relu(x)


class Tanh(BackendNode):
    def __call__(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.tanh(x)


class Sigmoid(BackendNode):
    """General Sigmoid Class."""

    def __call__(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.backend.library.sigmoid(x)
