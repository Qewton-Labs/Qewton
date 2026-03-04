import torch

from ..base import AlgorithmNode, AlgorithmState, AlgorithmAttributes
from ...config.variables import Variable
from ...optim.hyperparameter.base import (
    HyperParameter,
    DiscreteHyperparameter,
    CategoricalHyperparameter,
)


def _construct_fc_layers(
    hidden_layers: int,
    hidden_neurons: int,
    input_dim: int,
    output_dim: int,
    activation,
):
    layers = []
    if hidden_layers > 0:
        layers.append(torch.nn.Linear(input_dim, hidden_neurons))
        layers.append(activation)

        for _ in range(hidden_layers - 1):
            layers.append(
                torch.nn.Linear(
                    hidden_neurons,
                    hidden_neurons,
                )
            )
            layers.append(activation)

        layers.append(torch.nn.Linear(hidden_neurons, output_dim))
    else:
        layers.append(torch.nn.Linear(input_dim, output_dim))

    return layers


class TorchFCN(AlgorithmNode):

    def __init__(
        self,
        input_variable: Variable,
        output_variable: Variable,
        hidden_layers: int | DiscreteHyperparameter,
        hidden_neurons: int | DiscreteHyperparameter,
        activation_fn: torch.nn.Module | CategoricalHyperparameter = torch.nn.Tanh(),
        name: str = "TorchFCN",
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
        layers = _construct_fc_layers(
            self.hidden_layer.value,
            self.hidden_neurons.value,
            self.input_variable.dim,
            self.output_variable.dim,
            self.activation_fn.value,
        )

        self.model = torch.nn.Sequential(*layers)
        self._state = AlgorithmState.READY

    def _run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.state == AlgorithmState.UNINITIALIZED:
            self.setup()
        data = inputs[self.InputKeys.INPUT]  # type: ignore
        outdata = self.model(data)
        return {self.OutputKeys.OUTPUT: outdata}

    @property
    def trainable_parameters(self):
        if self.model is not None:
            return self.model.parameters()
        return None

    def to(self, device):
        """Move data stored in this node to a different device (GPU, CPU)"""
        if self.model is not None:
            self.model = self.model.to(device=device)

    def reset(self):
        if not self.state == AlgorithmState.FIXED:
            self.model: torch.nn.Module = torch.nn.Module()
            self._state = AlgorithmState.UNINITIALIZED

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self.hidden_layer, self.hidden_neurons, self.activation_fn]

    @property
    def attributes(self) -> set[AlgorithmAttributes]:
        return {
            AlgorithmAttributes.TRAINABLE,
            AlgorithmAttributes.DIFFERENTIABLE,
            AlgorithmAttributes.DETERMINISTIC,
            AlgorithmAttributes.GPU_ACCELERATED,
        }
