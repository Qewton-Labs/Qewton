import torch

from .base import AlgorithmNode, AlgorithmState
from ..config.variables import Variable
from ..optim.hyperparameter.base import (
    HyperParameter,
    DiscreteHyperparameter,
    CategoricalHyperparameter,
)


class TorchFCN(AlgorithmNode):

    def __init__(
        self,
        input_variable: Variable,
        output_variable: Variable,
        hidden_layers: int | DiscreteHyperparameter,
        hidden_neurons: int | DiscreteHyperparameter,
        activation_fn: torch.nn.Module | CategoricalHyperparameter = torch.nn.Tanh(),
        name: str = "TorchFCNNode",
    ) -> None:
        super().__init__(
            input_variable=input_variable, output_variable=output_variable, name=name
        )
        self.model: torch.nn.Module
        self.hidden_layer = HyperParameter.from_value(hidden_layers, "Hidden Layers")
        self.hidden_neurons = HyperParameter.from_value(hidden_neurons, "Hidden Neurons")
        self.activation_fn = HyperParameter.from_value(
            activation_fn, "Activation Function"
        )

    def setup(self) -> None:
        """Creates the underlying algorithm instance (e.g. creates the
        neural network)
        """
        layers = []
        if self.hidden_layer.current_value > 0:
            layers.append(
                torch.nn.Linear(
                    self.input_variable.dim, self.hidden_neurons.current_value
                )
            )
            layers.append(self.activation_fn.current_value)

            for _ in range(self.hidden_layer.current_value - 1):
                layers.append(
                    torch.nn.Linear(
                        self.hidden_neurons.current_value,
                        self.hidden_neurons.current_value,
                    )
                )
                layers.append(self.activation_fn.current_value)

            layers.append(
                torch.nn.Linear(
                    self.hidden_neurons.current_value, self.output_variable.dim
                )
            )
        else:
            layers.append(
                torch.nn.Linear(self.input_variable.dim, self.output_variable.dim)
            )

        self.model = torch.nn.Sequential(*layers)
        self._state = AlgorithmState.READY

    def run(
        self, inputs: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        if self.state == AlgorithmState.UNINITIALIZED:
            self.setup()
        data = inputs[self.InputKeys.INPUT]  # type: ignore
        outdata = self.model(data)
        return {self.OutputKeys.OUTPUT: outdata}

    @property
    def trainable_parameters(self):  # type: ignore
        return self.model.parameters()

    def to(self, device):
        """Move data stored in this node to a different device (GPU, CPU)"""
        self.model = self.model.to(device=device)

    def reset(self):
        if not self.state == AlgorithmState.FIXED:
            self.model: torch.nn.Module = torch.nn.Module()
            self._state = AlgorithmState.UNINITIALIZED

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self.hidden_layer, self.hidden_neurons, self.activation_fn]
