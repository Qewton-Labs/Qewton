from abc import abstractmethod
import copy

import numpy as np

from ...graphs.nodes import NodeState
from ...algorithms.backend import Backend, DEFAULT_DL_BACKEND
from ...config.variables import Variable

from ...optim.parameters.hyperparameter_base import HyperParameter
from ...optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
)
from ...optim.parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
)
from ...optim.base import EvaluationPhase
from ...graphs.nodes import Node, OutputPort, InputPort
from ..datasets import DataSet

# DATASET_REGISTRY = []


# # TODO: Is this a clean way to register child classes without importing them?
# def register_dataset(condition: Callable[..., bool], cls_type: type):
#     """Register a condition to choose the dataset + dataset class"""
#     DATASET_REGISTRY.append((condition, cls_type))


class DataNode(Node):
    def __init__(
        self, batch_size, name: str = "DataNode", state: NodeState = NodeState.FIXED
    ) -> None:
        self._batch_size = HyperParameter.from_value(batch_size, name="batch_size")
        self._batch_progress = 0
        self._is_cached = False
        self._device = None
        super().__init__(name, state)

    @property
    def batch_size(self) -> int:
        return self._batch_size.value

    @property
    def is_cached(self):
        return self._is_cached

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self._batch_size]

    @abstractmethod
    def cache(self, n_batches=-1):
        pass


class PointSampler(DataNode):
    pass


class DataLoader(DataNode):
    """
    TODO: parallelize this, similar to pytorch dataloader
    pin memory flag?
    """

    def __init__(
        self,
        data_set: DataSet,
        batch_size: int | DiscreteHyperparameter | CategoricalHyperparameter,
        splitting_ratio: tuple[float, float, float] = (1.0, 0.0, 0.0),
        shuffle_data: bool = True,
        shuffle_seed: int | None = None,
        backend: Backend | None = DEFAULT_DL_BACKEND,
        name: str = "DataLoader",
    ):
        self.data_set = data_set
        self.splitting_ratio = splitting_ratio
        assert sum(self.splitting_ratio) == 1.0, "Splitting ratio must sum to 1.0"
        self.shuffle_data = shuffle_data
        self.shuffle_seed = shuffle_seed

        if shuffle_data:
            if shuffle_seed is not None:
                rng = np.random.default_rng(self.shuffle_seed)
                self.permutation = rng.permutation(len(self.data_set))
            else:
                self.permutation = np.random.permutation(len(self.data_set))
        else:
            self.permutation = np.arange(len(self.data_set))

        self.backend = backend
        self._batch_progress = 0

        super().__init__(batch_size=batch_size, name=name)

        self._build_output_ports()
        self._build_data_split()

    def _build_output_ports(self):
        self._output_ports = []
        copy_memo = {}
        for config in self.data_set.data_configs:
            copied_config = copy.deepcopy(config, memo=copy_memo)
            _, dim = copied_config.get_axes_and_dim(0)  # TODO: assume 0 is batch axis?
            if dim is None:
                raise ValueError("Expected axis 0 to be a batch axis. But got None.")
            dim.update_size(self.batch_size)

            self._output_ports.append(
                OutputPort(
                    copied_config,
                    self,
                    name=config.variable_name,
                )
            )

    def _build_data_split(self):
        n_samples = len(self.data_set)
        train_r, val_r, _ = self.splitting_ratio
        train_end = int(train_r * n_samples)
        val_end = train_end + int(val_r * n_samples)
        self._splits = {
            EvaluationPhase.TRAIN: (0, train_end),
            EvaluationPhase.VALIDATION: (train_end, val_end),
            # TODO: Tuning checks on the validation data???
            EvaluationPhase.TUNE: (train_end, val_end),
            EvaluationPhase.TEST: (val_end, n_samples),
            EvaluationPhase.ALWAYS: (0, n_samples),
        }

    @property
    def input_ports(self) -> list[InputPort]:
        return []

    @property
    def output_ports(self) -> list[OutputPort]:
        return self._output_ports

    def set_mode(self, new_mode):
        if new_mode != self.mode:
            self._batch_progress = 0  # reset batch
        self.mode = new_mode

    def to(self, device):
        self._device = device

    def forward(self):
        start_split, end_split = self._splits[self.mode]

        start = start_split + self._batch_progress
        end = min(start + self.batch_size, end_split)

        current_idx = self.permutation[start:end]
        batch = self.data_set.get_batch(current_idx)
        if self._device is not None and self.backend is not None:
            batch = [self.backend.to(b, self._device) for b in batch]

        # update batch progress
        self._batch_progress += self.batch_size
        # reset if exceeding split
        if end >= end_split:
            self._batch_progress = 0

        return tuple(batch)

    def cache(self, n_batches=-1):
        return

    def get_output_port(self, name: str | Variable):
        if isinstance(name, str):
            return super().get_output_port(name)
        if isinstance(name, Variable):
            for port in self.output_ports:
                if port.name == name.name:
                    return port
        raise ValueError(f"No output port with name {name} found in node {self.name}.")
