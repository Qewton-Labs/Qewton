import math
from typing import Callable
from abc import abstractmethod

import numpy as np

from ...graphs.nodes import NodeState
from ...algorithms.backend import Backend

from ...config import DataConfiguration, AxesDim
from ...optim.base import EvaluationPhase
from ...optim.parameters.hyperparameter_base import HyperParameter
from ...optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
)
from ...optim.parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
)
from ...config.axes import BatchAxes
from ...graphs.nodes import Node, OutputPort, InputPort
from ..datasets import DataSet

# TODO: For now just a simple dataset where the data is provided
# How do we handle splitting the data for training, testing, validation?
# Currently everything is done here, but maybe split this further?
# With the current way we can use the same graph for multiple things,
# but this also makes it less transparent
#
# We need DataSets that can:
# - load data on the fly from a file/source
# - run other methods/software to create data


DATASET_REGISTRY = []


# TODO: Is this a clean way to register child classes without importing them?
def register_dataset(condition: Callable[..., bool], cls_type: type):
    """Register a condition to choose the dataset + dataset class"""
    DATASET_REGISTRY.append((condition, cls_type))


class DataNode(Node):
    """
    A Node that loads or samples data. Has only output ports in the graph,
    no input ports.
    """

    def __init__(
        self,
        batch_size: int | DiscreteHyperparameter | CategoricalHyperparameter,
        name: str = "DataNode",
        state: NodeState = NodeState.FIXED,
    ) -> None:
        self._batch_size = HyperParameter.from_value(batch_size)
        self._is_cached = False
        super().__init__(name, state)

    @property
    def batch_size(self):
        return self._batch_size.value

    @abstractmethod
    def __len__(self):
        pass

    @property
    def is_cached(self):
        return self._is_cached

    @abstractmethod
    def cache(self, n_batches=-1):
        pass


class PointSampler(DataNode):
    pass


class DataLoader(DataNode):
    """
    TODO: parallelize this, similar to pytorch dataloader
    pin memory flag?

    Implements a standard dataloader module.


    """

    def __init__(
        self,
        data_set: DataSet,
        batch_size: int | DiscreteHyperparameter | CategoricalHyperparameter,
        splitting_ratio: tuple[float, float, float] = (1.0, 0.0, 0.0),
        shuffle_data: bool | CategoricalHyperparameter = True,
        shuffle_seed: int | None = None,
        backend: Backend | None = None,
        name: str = "DataLoader",
    ):
        self.data_set = data_set
        self.splitting_ratio = splitting_ratio
        self.shuffle_data = HyperParameter.from_value(shuffle_data)
        self.shuffle_seed = shuffle_seed
        self._rng = np.random.default_rng(self.shuffle_seed)

        self._batch_progress = 0
        self.permutation = []
        self._permutation_splits = {}
        self.setup_iteration()

        self.backend = backend

        super().__init__(batch_size=batch_size, name=name)

        self._output_ports = []
        for config in self.data_set.data_configs:
            axes = list(config.axes)  # TODO: hier lieber deep kopieren oder nicht?
            assert isinstance(axes[0], BatchAxes), "In DataSets, \
                the first axes should be the batch axes."
            assert len(axes[0].shape) == 1, "Multi-dimensional \
                batch axes not supported for batching."
            assert axes[0].shape[0] >= self.batch_size, "Batch \
                can not be larger than dataset size."
            axes[0] = BatchAxes(AxesDim(self.batch_size))
            new_config = DataConfiguration(
                *axes, dtype=backend.standard_datatype() if backend else None
            )
            port_name = str(
                config.feature_axes.variables if config.feature_axes else None
            )

            self._output_ports.append(
                OutputPort(
                    new_config,
                    self,
                    name=port_name,
                )
            )

    def set_permutation(self):
        if self.shuffle_data:
            self.permutation = self._rng.permutation(len(self.data_set))
        else:
            self.permutation = np.arange(len(self.data_set))

    def setup_iteration(self):
        self._batch_progress = 0
        self.set_permutation()
        n_samples = len(self.permutation)
        r_train, r_val, _ = self.splitting_ratio

        train_end = int(r_train * n_samples)
        val_end = train_end + int(r_val * n_samples)

        self._permutation_splits = {
            EvaluationPhase.TRAIN: self.permutation[0:train_end],
            EvaluationPhase.VALIDATION: self.permutation[train_end:val_end],
            EvaluationPhase.TUNE: self.permutation[train_end:val_end],
            EvaluationPhase.TEST: self.permutation[val_end:n_samples],
            EvaluationPhase.ALWAYS: self.permutation[0:n_samples],
        }

    def __len__(self):
        return math.ceil(len(self.data_set) / self.batch_size)

    def run(self):
        split_indices = self._permutation_splits[self.mode]
        n_split = len(split_indices)

        if n_split == 0:
            return

        bs = self.batch_size
        if hasattr(bs, "value"):
            bs = bs.value

        if self._batch_progress >= n_split:
            self._batch_progress = 0
            if self.shuffle_data:
                self._rng.shuffle(split_indices)

        indices = split_indices[self._batch_progress : self._batch_progress + bs]
        batch_data = self.data_set.get_batch(indices)

        self._batch_progress += bs

        for i, data in enumerate(batch_data):
            self.output_ports[i].set_value(data)

    @property
    def input_ports(self) -> list[InputPort]:
        return []

    @property
    def output_ports(self) -> list[OutputPort]:
        return self._output_ports

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self.batch_size, self.shuffle_data]

    def set_mode(self, new_mode):
        if new_mode != self.mode:
            self._batch_progress = 0  # reset batch
        self.mode = new_mode

    def to(self, device):
        pass
