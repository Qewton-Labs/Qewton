from typing import Annotated

import tensorflow as tf
import torch

from qewton.algorithms.backend_node import BackendNode, TensorType
from qewton.config.axes import EllipsisAxes
from qewton.config.data_configurations import DataConfiguration


class ReIm(BackendNode[TensorType]):
    ellipsis_axes = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DataConfiguration(ellipsis_axes)]
    ) -> tuple[
        Annotated[TensorType, DataConfiguration(ellipsis_axes)],
        Annotated[TensorType, DataConfiguration(ellipsis_axes)],
    ]:
        return self.implementation(x)

    def torch_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x.real, x.imag

    def tensorflow_implementation(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return self.backend.library.math.real(x), self.backend.library.math.imag(x)
