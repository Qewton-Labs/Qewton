from __future__ import annotations
from typing import Any
import torch

from pioneer.config.variables import Variable

from ...config.configuration_base import DataConfiguration
from ...optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
)
from ...optim.parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
)
from ...optim.base import EvaluationPhase
from .base import DataSet, register_dataset


class TorchDataSet(DataSet):

    def __init__(
        self,
        data_config: DataConfiguration,
        data: torch.Tensor,
        batch_size: int | DiscreteHyperparameter | CategoricalHyperparameter,
        batch_dimension: int = 0,
        splitting_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        shuffle_data: bool = True,
        name: str = "TorchDataSet",
    ):
        assert isinstance(
            data, torch.Tensor
        ), "TorchDataSet expects torch tensors as data."
        super().__init__(
            data_config,
            data,
            batch_size,
            batch_dimension=batch_dimension,
            splitting_ratio=splitting_ratio,
            shuffle_data=shuffle_data,
            name=name,
        )
        self.data: torch.Tensor = self.data

        if self.shuffle_data:
            rand_idx = torch.randperm(self.data.size(self.batch_dimension))
            self.data = self.data[self.build_axis_slice(self.batch_dimension, rand_idx)]

        # Build data splits for batching and different modes
        n_samples = self.data.size(self.batch_dimension)
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
        self._mean = self.data.mean(dim=self.batch_dimension, keepdim=True)

    def _compute_std(self):
        self._std = self.data.std(dim=self.batch_dimension, keepdim=True)

    def run(self):
        start_split, end_split = self._splits[self.mode]

        start = start_split + self._batch_progress
        end = min(start + self.batch_size.value, end_split)

        batch = self.data[self.build_axis_slice(self.batch_dimension, slice(start, end))]

        # update batch progress
        self._batch_progress += self.batch_size.value
        # reset if exceeding split
        if end >= end_split:
            self._batch_progress = 0

        self.output.set_value(batch)

    def reset(self):
        self.to("cpu")

    def compute_pca(self, n_components: int, variable: Variable) -> tuple[Any, Any, Any]:
        # TODO: Implement
        return (None, None, None)


register_dataset(lambda d: isinstance(d, torch.Tensor), TorchDataSet)
