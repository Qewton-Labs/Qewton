from __future__ import annotations
from typing import Any, Annotated

from qewton.backends import DEFAULT_DL_BACKEND
from qewton.backends.base import DeepLearningBackend, TensorType
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.trainable_parameters import TrainableParameters
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import EllipsisAxes, FeatureAxes
from qewton.graphs.nodes import Node, NodeState, NodeConfig


class ParameterNode(Node[TensorType]):
    _type_identifier = "ParameterNode"

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

    def set_trainable_parameter(self, new_value: TensorType) -> None:
        if self.state == NodeState.FIXED:
            raise ValueError("Cannot set trainable parameter when node is fixed.")
        self._trainable_parameter = new_value
        self.output.set_value(self._trainable_parameter)

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

    def config_dict(self) -> NodeConfig:
        other_args = {
            "state": self.state,
            "name": self.name,
            "initial_value": self.initial_value,
            "backend": self.backend,
            "trainable_parameters": self.trainable_parameters,
        }
        hyperparameters = {}
        for s in self.shape:
            hyperparameters[s.name] = s

        return NodeConfig(
            node_identifier=ParameterNode._type_identifier,
            node_id=self.node_id,
            mode=self.mode,
            hyperparameters=hyperparameters,
            other_args=other_args,
            state=self.state,
        )

    @classmethod
    def load_from_config(cls, config: NodeConfig) -> Node:
        """Reconstructs a node from a configuration object. By default we just
        use the hyperparameters and other arguments, but this can be overridden
        in subclasses to include additional information.

        Args:
            config (NodeConfig): The configuration object.

        Returns:
            Node: The reconstructed node.
        """
        node: ParameterNode = ParameterNode(
            shape=tuple(s for s in config.hyperparameters.values()),
            name=config.other_args.get("name", "ParameterNode"),
            initial_value=config.other_args.get("initial_value", None),
            backend=config.other_args.get("backend", DEFAULT_DL_BACKEND),
        )
        node.set_mode(config.mode)
        node.node_id = config.node_id
        node.set_trainable_parameter(config.other_args.get("trainable_parameters", None))
        node.set_state(config.state)
        return node
