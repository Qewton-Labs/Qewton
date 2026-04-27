from typing import Any, Annotated

from ..backend import DEFAULT_DL_BACKEND
from ...config.configuration_base import DataConfiguration
from ..backend_node import BackendNode

from ...graphs.nodes import NO_DEFAULT


class Narrow(BackendNode):
    def __init__(self, dim=None, start=0, length=None, backend=DEFAULT_DL_BACKEND):
        self.dim = dim if dim is not None else NO_DEFAULT
        self.start = start
        self.length = length if length is not None else NO_DEFAULT
        super().__init__(name=None, backend=backend)

    def forward(
        self,
        x: Annotated[Any, DataConfiguration([])],
    ) -> Annotated[Any, DataConfiguration([])]:
        return self.implementation(x)

    def torch_implementation(self, x):
        return self.backend.library.narrow(x, self.dim, self.start, self.length)
