from typing import Literal

from qewton.graphs.nodes import Node
from qewton.optim.trainer.training_controllers import TrainerState
from qewton.optim.trainer.callbacks.base_callback import Callback


class ModelParameterSizeCallback(Callback):
    """Saves the size of the model in at the end of the training process.

    Args:
        parameter_nodes (list[Node]):
            List of model parameter nodes to calculate the size of.
        memory_format (Literal["KB", "MB", "GB"], optional):
            Memory format to use for the size. Defaults to "MB".
        priority (int, optional): Priority of the callback.
            Callbacks with higher priority are executed first. Defaults to 0.
    """

    def __init__(
        self,
        parameter_nodes: list[Node],
        memory_format: Literal["KB", "MB", "GB"] = "MB",
        priority=0,
    ) -> None:
        self.parameter_nodes = parameter_nodes
        self.memory_key = f"Parameter size [{memory_format}]"
        self.memory_scale = 1024 ** {"KB": 1, "MB": 2, "GB": 3}[memory_format]
        super().__init__(priority)

    def add_callback_info_to_state(self, state: TrainerState):
        state.callback_info[self.memory_key] = 0

    def on_training_start(self, state: TrainerState):
        total_size = 0
        for node in self.parameter_nodes:
            total_size += node.memory_consumption
        state.callback_info[self.memory_key] = round(total_size / self.memory_scale, 2)
