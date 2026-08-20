from __future__ import annotations
from typing import Any, Annotated

from qewton.backends import DEFAULT_DL_BACKEND
from qewton.backends.base import DeepLearningBackend, TensorType
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.trainable_parameters import TrainableParameters
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import EllipsisAxes, FeatureAxes
from qewton.graphs.nodes import Node, NodeState


class ParameterNode(Node[TensorType]):
    def __init__(
        self,
        shape: tuple[int | HyperParameter, ...],
        initial_value=None,
        name: str = "ParameterNode",
        backend: type[DeepLearningBackend] = DEFAULT_DL_BACKEND,
    ) -> None:
        self.shape = tuple(
            HyperParameter.from_value(s, f"shape_{i}") for i, s in enumerate(shape)
        )
        self._trainable_parameter: Any | None = None
        self.initial_value = initial_value
        self.backend: DeepLearningBackend
        super().__init__(name, state=NodeState.UNINITIALIZED, backend=backend)
        self.output = self.output_ports[0]

    def setup(self) -> None:
        if self.state == NodeState.UNINITIALIZED:
            if self.initial_value is not None:
                if not hasattr(self.initial_value, "shape"):
                    raise ValueError(f"initial_value must have a 'shape' attribute,\
                            got {type(self.initial_value)}")
                self._trainable_parameter = self.backend.param.initialize(
                    self.initial_value.shape, self.initial_value
                )
            else:
                int_shape = tuple(hp.value for hp in self.shape)
                self._trainable_parameter = self.backend.param.initialize(int_shape)
            self.output.set_value(self._trainable_parameter)
            self._state = NodeState.INITIALIZED

    def output_config(self):
        int_shape = tuple(hp.value for hp in self.shape)
        return DataConfiguration(
            EllipsisAxes(),
            FeatureAxes(shape=int_shape),
            dtype=self.backend.default_dtype,  # type: ignore
        )

    def run(self) -> None:
        pass  # value is set once in setup

    def forward(self) -> Annotated[TensorType, ParameterNode.output_config]:
        return self._trainable_parameter  # type: ignore

    def reset(self):
        if not self.state == NodeState.FIXED:
            self.output.reset_value()
            self._state = NodeState.UNINITIALIZED
            self._trainable_parameter = None

    def fix_node_state(self) -> None:
        if not self._state == NodeState.UNINITIALIZED:
            self._trainable_parameter = self.backend.param.requires_grad(
                self._trainable_parameter, False
            )
            self.output.set_value(self._trainable_parameter)
        super().fix_node_state()

    def set_state(self, new_state: NodeState):
        super().set_state(new_state)
        if new_state == NodeState.FIXED:
            self.fix_node_state()

    @property
    def trainable_parameters(self):
        if self.state == NodeState.FIXED:
            params = []
        else:
            params = self._trainable_parameter
        return TrainableParameters(self.node_id, params)

    def to(self, device):
        if not self._state == NodeState.UNINITIALIZED:
            self._trainable_parameter = self.backend.param.to(
                self._trainable_parameter, device=device
            )
            self.output.set_value(self._trainable_parameter)
