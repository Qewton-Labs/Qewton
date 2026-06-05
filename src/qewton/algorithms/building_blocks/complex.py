from typing import Annotated

import tensorflow as tf
import torch

from qewton.algorithms.backend_node import BackendNode, TensorType
from qewton.algorithms.building_blocks.utility import Cast
from qewton.config.axes import EllipsisAxes
from qewton.config.backend import Backend
from qewton.config.data_configurations import DataConfiguration
from qewton.graphs.control_nodes.graph_node import CopiedNode
from qewton.graphs.nodes import Node


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


class ComplexValued(TrackedNode):
    fr: CopiedNode
    fi: CopiedNode
    fr2: CopiedNode
    fi2: CopiedNode
    reim: ReIm
    cast: Cast
    cast2: Cast

    def __init__(
        self,
        module: Node,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
        name: str = "complex_valued",
    ):
        self.fr = CopiedNode(module)
        self.fi = CopiedNode(module)
        self.fr2 = CopiedNode(module)
        self.fi2 = CopiedNode(module)
        self.reim = ReIm(backend=backend)
        self.cast = Cast(backend=backend)
        self.cast2 = CopiedNode(self.cast)
        super().__init__(name, backend)

    def forward(
        self, x: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        re, im = self.reim(x)

        return self.cast2(
            self.cast(
                self.fr(re) - self.fi(im),
                "complex64",
            )
            + 1j * (self.fr2(im) + self.fi2(re)),
            "complex64",
        )
