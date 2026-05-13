from typing import Any, Callable, Annotated
from abc import abstractmethod
import copy

import numpy as np

from ...graphs.nodes import NodeState
from ...algorithms.backend import Backend

from ...config import DataConfiguration, AxesDim
from ...optim.parameters.hyperparameter_base import HyperParameter
from ...optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
)
from ...optim.parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
)
from ...config.variables import Variable
from ...config.axes import BatchAxes, EllipsisAxes
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
    def __init__(
        self, batch_size, name: str = "DataNode", state: NodeState = NodeState.FIXED
    ) -> None:
        self._batch_size = batch_size
        self._is_cached = False
        super().__init__(name, state)

    @property
    def batch_size(self):
        return self._batch_size

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
    """

    def __init__(
        self,
        data_set: DataSet,
        batch_size: int | DiscreteHyperparameter | CategoricalHyperparameter,
        splitting_ratio: tuple[float, float, float] = (1.0, 0.0, 0.0),
        shuffle_data: bool = True,
        shuffle_seed: int | None = None,
        backend: Backend | None = None,
        name: str = "DataLoader",
    ):
        self.data_set = data_set
        self.splitting_ratio = splitting_ratio
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

        super().__init__(batch_size=batch_size, name=name)

        batch_config = DataConfiguration(
            BatchAxes(AxesDim(self.batch_size)), EllipsisAxes()
        )
        self._output_ports = []
        for config in self.data_set.data_configs:
            copied_config = copy.deepcopy(config)
            _, unified = batch_config.unify_with(copied_config)
            copied_config.update_config(unified)

            self._output_ports.append(
                OutputPort(
                    copied_config,
                    self,
                    name=config.variable_name,
                )
            )

    @property
    def input_ports(self) -> list[InputPort]:
        return []

    @property
    def output_ports(self) -> list[OutputPort]:
        return self._output_ports

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self.batch_size]

    def set_mode(self, new_mode):
        if new_mode != self.mode:
            self._batch_progress = 0  # reset batch
        self.mode = new_mode

    def to(self, device):
        pass
