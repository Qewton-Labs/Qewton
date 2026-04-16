from __future__ import annotations
from typing import Any
from ..implementation import (
    DEFAULT_DL_IMPLEMENTATION,
    TorchImplementation,
)
from ...optim.parameters.hyperparameter_base import HyperParameter
from ...optim.parameters.trainable_parameters import TrainableParameters
from ...config.configuration_base import DataConfiguration
from ...graphs.nodes import Node, NodeState, OutputPort


class _InternalParameter:

    def __init__(self, shape) -> None:
        self.shape = shape

    def to(self, device):
        pass

    @property
    def trainable_parameters(self) -> Any:
        pass

    def requires_grad(self, requires_grad: bool):
        pass


class TorchParameter(_InternalParameter):

    def __init__(self, shape=None, tensor=None) -> None:
        import torch  # type: ignore

        if tensor is not None:
            assert isinstance(tensor, torch.Tensor)
            self.param = torch.nn.Parameter(tensor)
        elif shape is not None:
            # TODO: We need some kind of initialization for these parameters
            # E.g. 0, rand, xavier,... But this also needs to be exposed to the outside
            self.param = torch.nn.Parameter(torch.zeros(shape), requires_grad=True)
            if len(shape) > 1:
                torch.nn.init.xavier_uniform_(self.param)
        super().__init__(shape)

    def to(self, device):
        self.param.to(device)

    @property
    def trainable_parameters(self):
        return self.param

    def requires_grad(self, requires_grad: bool):
        self.param.requires_grad = requires_grad


class ParameterNode(Node):

    existing_implementations = {TorchImplementation: TorchParameter}

    def __init__(
        self,
        shape: tuple[int | HyperParameter, ...],
        initial_value=None,
        name: str = "ParameterNode",
        backend=DEFAULT_DL_IMPLEMENTATION,
    ) -> None:
        super().__init__(name, state=NodeState.UNINITIALIZED)
        self.shape = tuple(
            HyperParameter.from_value(s, f"shape_{i}") for i, s in enumerate(shape)
        )
        self.backend = backend
        self.implementation_class = self.existing_implementations[self.backend]
        self.implementation: _InternalParameter | None = None
        self.initial_value = initial_value
        self.output = OutputPort(
            data_configuration=DataConfiguration([]),
            node=self,
            name="parameters",
        )

    def setup(self) -> None:
        if self.state == NodeState.UNINITIALIZED:
            if self.initial_value is not None:
                self.implementation = self.implementation_class(
                    self.initial_value.shape, self.initial_value
                )
            else:
                int_shape = tuple(hp.value for hp in self.shape)
                self.implementation = self.implementation_class(int_shape)
            self.output.set_value(self.implementation.param)
            self._state = NodeState.INITIALIZED

    def run(self) -> None:
        pass

    def reset(self):
        if not self.state == NodeState.FIXED:
            self.output.reset_value()
            self._state = NodeState.UNINITIALIZED

    def fix_node_state(self) -> None:
        super().fix_node_state()
        if self.implementation is not None:
            self.implementation.requires_grad(False)

    def set_state(self, new_state: NodeState):
        super().set_state(new_state)
        if new_state == NodeState.FIXED:
            self.fix_node_state()

    @property
    def trainable_parameters(self):
        params = (
            []
            if self.implementation is None
            else self.implementation.trainable_parameters
        )
        return TrainableParameters(self.node_id, params)

    def to(self, device):
        if self.implementation is not None:
            self.implementation.to(device=device)
