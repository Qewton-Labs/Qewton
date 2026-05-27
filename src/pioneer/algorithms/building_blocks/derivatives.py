from typing import Annotated

from pioneer.config.backend import DEFAULT_DL_BACKEND, Backend

from ...config.axes import EllipsisAxes, FeatureAxes
from ...config.backend import DEFAULT_DL_BACKEND, TensorType
from ...config.data_configurations import DataConfiguration
from ..backend_node import BackendNode

from ...graphs.nodes import NO_DEFAULT


class GradientTracking(BackendNode[TensorType]):
    ell_axes = EllipsisAxes()

    def forward(
        self,
        input: Annotated[TensorType, DataConfiguration(ell_axes)],
    ) -> Annotated[TensorType, DataConfiguration(ell_axes)]:
        return self.implementation(input)

    def torch_implementation(self, input):
        input.requires_grad = True
        return input


class Gradient(BackendNode[TensorType]):
    # TODO: Make config coupling better (input should be scalar)
    ell_axes = EllipsisAxes()

    def forward(
        self,
        u: Annotated[
            TensorType,
            DataConfiguration(EllipsisAxes()),
        ],
        x: Annotated[TensorType, DataConfiguration(ell_axes)],
    ) -> Annotated[TensorType, DataConfiguration(ell_axes)]:
        return self.implementation(u, x)

    def torch_implementation(self, u, x):
        return self.backend.library.autograd.grad(u.sum(), x, create_graph=True)[0]
