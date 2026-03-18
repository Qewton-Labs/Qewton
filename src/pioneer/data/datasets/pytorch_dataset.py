from __future__ import annotations
import torch

from ...config.configuration_base import DataConfiguration
from ...optim.hyperparameter.number_hyperparameter import (
    DiscreteHyperparameter,
)
from ...optim.hyperparameter.categorical_hyperparameter import (
    CategoricalHyperparameter,
)
from ...optim.base import EvaluationPhase
from ...config.variables import Variable
from .base import DataSet, register_dataset


class TorchDataSet(DataSet):

    def __init__(
        self,
        data_config: DataConfiguration,
        data,
        batch_size: int | DiscreteHyperparameter | CategoricalHyperparameter,
        splitting_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        shuffle_data: bool = True,
        name: str = "TorchDataSet",
    ):
        assert isinstance(
            data, torch.Tensor
        ), "TorchDataSet expects torch tensors as data."
        assert isinstance(
            data_config.dtype, torch.dtype
        ), f"Type of configuration is {data_config.dtype} and does not fit torch data."
        super().__init__(
            data_config,
            data,
            batch_size,
            splitting_ratio,
            shuffle_data=shuffle_data,
            name=name,
        )
        self.data: torch.Tensor = self.data

        if self.shuffle_data:
            rand_idx = torch.randperm(self.data.size(self.data_config.batch_axis_idx))
            self.data = self.data[
                self.data_config.slice_axis(self.data_config.batch_axis_idx, rand_idx)
            ]

        # Build data splits for batching and different modes
        n_samples = self.data.size(self.data_config.batch_axis_idx)
        train_r, val_r, _ = self.splitting_ratio
        train_end = int(train_r * n_samples)
        val_end = train_end + int(val_r * n_samples)
        self._splits = {
            EvaluationPhase.TRAIN: (0, train_end),
            EvaluationPhase.VALIDATION: (train_end, val_end),
            EvaluationPhase.TUNE: (
                train_end,
                val_end,
            ),  # TODO: Tuning checks on the validation data???
            EvaluationPhase.TEST: (val_end, n_samples),
            EvaluationPhase.ALWAYS: (0, n_samples),
        }

    def to(self, device):
        self.data = self.data.to(device)
        if self._mean is not None:
            self._mean = self._mean.to(device)
        if self._std is not None:
            self._std = self._std.to(device)

    def _compute_mean(self):
        self._mean = self.data.mean(dim=self.data_config.batch_axis_idx, keepdim=True)

    def _compute_std(self):
        self._std = self.data.std(dim=self.data_config.batch_axis_idx, keepdim=True)

    def run(self):
        start_split, end_split = self._splits[self.mode]

        start = start_split + self._batch_progress
        end = min(start + self.batch_size.value, end_split)

        batch = self.data[
            self.data_config.slice_axis(
                self.data_config.batch_axis_idx, slice(start, end)
            )
        ]

        # update batch progress
        self._batch_progress += self.batch_size.value
        # reset if exceeding split
        if end >= end_split:
            self._batch_progress = 0

        self.out_port.set_value(batch)

    def compute_pca(
        self, n_components: int, variable: Variable
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_data = (self.data - self.mean) / (self.std + self.std_eps)
        variable_idx = self.data_config.get_axis_indices_of_variables(variable)
        index_slice = self.data_config.slice_axis(
            self.data_config.feature_axis_idx, variable_idx
        )
        return torch.pca_lowrank(
            torch.flatten(normalized_data[index_slice], 1), q=n_components
        )

    def reset(self):
        self.to("cpu")


register_dataset(lambda d: isinstance(d, torch.Tensor), TorchDataSet)
