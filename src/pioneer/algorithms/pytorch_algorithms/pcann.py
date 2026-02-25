import torch

from .fcn import _construct_fc_layers
from ..base import AlgorithmNode, AlgorithmState, AlgorithmAttributes
from ...nodes.base import Port
from ...data.datasets.base import DataSet
from ...config.variables import Variable
from ...optim.hyperparameter.base import (
    HyperParameter,
    DiscreteHyperparameter,
    CategoricalHyperparameter,
)


# TODO: Just some test version!
class TorchPCANN(AlgorithmNode):
    def __init__(
        self,
        input_variable: Variable,
        output_variable: Variable,
        input_dataset: DataSet,
        output_dataset: DataSet,
        input_pca_components: int | DiscreteHyperparameter,
        output_pca_components: int | DiscreteHyperparameter,
        hidden_layers: int | DiscreteHyperparameter,
        hidden_neurons: int | DiscreteHyperparameter,
        activation_fn: torch.nn.Module | CategoricalHyperparameter = torch.nn.Tanh(),
        name: str = "TorchPCANN",
    ) -> None:
        super().__init__(
            input_variable=input_variable, output_variable=output_variable, name=name
        )
        self.model: torch.nn.Module
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self.in_port = Port(
            self.input_dataset.data_config[self.input_variable], self, "input", True
        )
        self.out_port = Port(
            self.output_dataset.data_config[self.output_variable], self, "output", True
        )
        # All hyperparameters:
        self.hidden_layer = HyperParameter.from_value(hidden_layers, "Hidden Layers")
        self.hidden_neurons = HyperParameter.from_value(hidden_neurons, "Hidden Neurons")
        self.activation_fn = HyperParameter.from_value(
            activation_fn, "Activation Function"
        )
        self.input_pca_components = HyperParameter.from_value(
            input_pca_components, "PCA components input"
        )
        self.output_pca_components = HyperParameter.from_value(
            output_pca_components, "PCA components output"
        )

    def setup(self) -> None:
        layers = _construct_fc_layers(
            self.hidden_layer.value,
            self.hidden_neurons.value,
            self.input_pca_components.value,
            self.output_pca_components.value,
            self.activation_fn.value,
        )
        self.model = torch.nn.Sequential(*layers)
        # get mean and std from dataset
        input_idx = self.input_dataset.data_config.get_axis_indices_of_variables(
            self.input_variable
        )
        output_idx = self.output_dataset.data_config.get_axis_indices_of_variables(
            self.output_variable
        )
        self._register_mean_and_std(self.input_dataset, input_idx, "mean_in", "std_in")
        self._register_mean_and_std(
            self.output_dataset, output_idx, "mean_out", "std_out"
        )

        # compute pca (TODO: currently only done for the components needed,
        # keep it like this, or compute everything once and then slice?)
        pca_in = self.input_dataset.compute_pca(
            self.input_pca_components.value, self.input_variable
        )
        pca_out = self.output_dataset.compute_pca(
            self.output_pca_components.value, self.output_variable
        )
        self.model.register_buffer("eigenvectors_in", pca_in[2])
        self.model.register_buffer("eigenvectors_out", pca_out[2])

        ev_values_in = torch.sqrt(pca_in[1] ** 2 / (len(pca_in[0]) - 1))
        self.model.register_buffer("eigenvalues_in", ev_values_in)
        ev_values_out = torch.sqrt(pca_out[1] ** 2 / (len(pca_out[0]) - 1))
        self.model.register_buffer("eigenvalues_out", ev_values_out)
        self._state = AlgorithmState.READY

    def _register_mean_and_std(
        self, dataset: DataSet, indices, mean_str: str, std_str: str
    ):
        index_slice = dataset.data_config.slice_axis(
            dataset.data_config.feature_axis_idx, indices
        )
        data_mean = dataset.mean
        data_std = dataset.std
        self.model.register_buffer(mean_str, data_mean[index_slice])
        self.model.register_buffer(std_str, data_std[index_slice])

    def run(
        self, inputs: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        if self.state == AlgorithmState.UNINITIALIZED:
            self.setup()
        data = inputs[self.InputKeys.INPUT]  # type: ignore
        # normalize inputs
        points = (data - self.model.mean_in) / self.model.std_in  # type: ignore
        # apply pca
        points = torch.flatten(points, start_dim=1)
        pc_in = points @ self.model.eigenvectors_in  # type: ignore
        pc_in /= self.model.eigenvalues_in  # type: ignore
        # Then evaluate neural network
        pc_out = self.model(pc_in)
        # "inverse" pca
        pc_out *= self.model.eigenvalues_out
        points_out = pc_out @ self.model.eigenvectors_out.T
        batch_size = points_out.shape[0]
        points_out = points_out.reshape(
            (batch_size, *self.model.std_out.shape[1:])  # type: ignore
        )
        # "inverse" normalization
        points_out = points_out * self.model.std_out + self.model.mean_out
        return {self.OutputKeys.OUTPUT: points_out}

    @property
    def trainable_parameters(self):
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
        return [
            self.hidden_layer,
            self.hidden_neurons,
            self.activation_fn,
            self.input_pca_components,
            self.output_pca_components,
        ]

    @property
    def input_ports(self) -> dict[str, Port]:
        return {self.InputKeys.INPUT: self.in_port}

    @property
    def output_ports(self) -> dict[str, Port]:
        return {self.OutputKeys.OUTPUT: self.out_port}

    @property
    def attributes(self) -> set[AlgorithmAttributes]:
        return {
            AlgorithmAttributes.TRAINABLE,
            AlgorithmAttributes.NORMALIZES_DATA,
            AlgorithmAttributes.DETERMINISTIC,
            AlgorithmAttributes.GPU_ACCELERATED,
        }
