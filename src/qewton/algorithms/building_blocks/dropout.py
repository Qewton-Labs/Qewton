from typing import Annotated, final, override

from qewton.algorithms.backend_node import BackendNode, TensorType
from qewton.config.axes import EllipsisAxes
from qewton.config.backend import (
    DEFAULT_DL_BACKEND,
    Backend,
    TensorflowBackend,
    TorchBackend,
)
from qewton.config.data_configurations import DataConfiguration
from qewton.optim.base import EvaluationPhase


@final
class Dropout(BackendNode[TensorType]):
    ellipsis_axes = EllipsisAxes()

    def __init__(
        self,
        p: float = 0.5,
        name: str = "dropout",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
        mode: EvaluationPhase = EvaluationPhase.TRAIN,
    ) -> None:
        super().__init__(name, backend)

        if self.backend == TorchBackend:
            ctr = self.backend.library.nn.Dropout
        elif self.backend == TensorflowBackend:
            ctr = self.backend.library.keras.layers.Dropout
        else:
            raise NotImplementedError

        self.implementation = ctr(p)
        self.set_mode(mode)

    @override
    def forward(
        self, x: Annotated[TensorType, DataConfiguration(ellipsis_axes)]
    ) -> Annotated[TensorType, DataConfiguration(ellipsis_axes)]:
        is_training = self.mode == EvaluationPhase.TRAIN
        if self.backend == TensorflowBackend:
            return self.implementation(x, training=is_training)
        elif self.backend == TorchBackend:
            if is_training:
                self.implementation.train()
            else:
                self.implementation.eval()
            return self.implementation(x)
        else:
            raise NotImplementedError
