"""
Base classes for data loading and node-based data sampling in the graph.
"""

from copy import deepcopy
import math
from abc import abstractmethod

import numpy as np

from qewton.graphs.nodes import NodeState
from qewton.backends import Backend, DEFAULT_DL_BACKEND, DeepLearningBackend
from qewton.config.variables import Variable

from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
)
from qewton.optim.parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
)
from qewton.config.axes import BatchAxes, AxesDim
from qewton.config.data_configurations import DataConfiguration
from qewton.graphs.nodes import Node, OutputPort, InputPort
from qewton.data.datasets import DataSet


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
        backend: type[Backend] | None = DEFAULT_DL_BACKEND,
    ) -> None:
        """
        Args:
            batch_size: Number of samples per batch.
            name: Display name of the node.
            state: Initial state of the node.
            backend: Computing backend (e.g., TorchBackend).
        """
        self._batch_size = HyperParameter.from_value(batch_size, name="batch_size")
        self._batch_progress = 0
        self._is_cached = False
        self._device = None
        if backend is None:
            backend = Backend
        super().__init__(name, state, backend=backend)

    @property
    def batch_size(self) -> int:
        return self._batch_size.value

    @abstractmethod
    def __len__(self):
        pass

    @property
    def is_cached(self):
        return self._is_cached

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self._batch_size]

    def cache(self, n_batches=-1):
        pass

    def provides_data_in_phase(self, phase: EvaluationPhase) -> bool:
        return False

    def to(self, device):
        self._device = device


class DataLoader(DataNode):
    """Standard DataLoader module for batching, shuffling, and splitting datasets.

    This node acts as a source in the computation graph, providing batches of
    data to connected algorithms.
    """

    # TODO: parallelize this, similar to pytorch dataloader
    # pin memory flag?

    def __init__(
        self,
        data_set: DataSet,
        batch_size: int | DiscreteHyperparameter | CategoricalHyperparameter,
        splitting_ratio: tuple[float, float, float] = (1.0, 0.0, 0.0),
        shuffle_data: bool | CategoricalHyperparameter = True,
        shuffle_seed: int | None = None,
        backend: type[Backend] = DEFAULT_DL_BACKEND,
        name: str = "DataLoader",
    ):
        """
        Args:
            data_set (DataSet): The source dataset.
            batch_size: Number of samples per batch.
            splitting_ratio: Proportions for (Train, Validation, Test) splits.
            shuffle_data: Whether to shuffle the indices at the start of an epoch.
            shuffle_seed: Random seed for reproducibility.
            backend: The backend used for data types and device transfers.
            name: Node name.
        """
        self.data_set = data_set
        self.splitting_ratio = splitting_ratio
        self.shuffle_data = HyperParameter.from_value(shuffle_data, "shuffle_data")
        self.shuffle_seed = shuffle_seed
        self._rng = np.random.default_rng(self.shuffle_seed)
        self.permutation = []
        self._permutation_splits = {}
        self.setup_iteration()

        super().__init__(batch_size=batch_size, name=name, backend=backend)

        # Build output ports based on dataset configurations
        self._output_ports = []
        copy_memo = {}
        for config in self.data_set.data_configs:
            axes = deepcopy(list(config.axes), memo=copy_memo)
            assert isinstance(axes[0], BatchAxes), "In DataSets, \
                the first axes should be the batch axes."
            assert len(axes[0].shape) == 1, "Multi-dimensional \
                batch axes not supported for batching."
            assert (
                axes[0].shape[0].size >= self.batch_size
            ), "Batch can not be larger than dataset size."
            axes[0] = BatchAxes(AxesDim(self.batch_size))
            new_config = DataConfiguration(
                *axes, dtype=backend.default_dtype if backend else None
            )
            self._output_ports.append(
                OutputPort(
                    new_config,
                    self,
                    name=config.variable_name,
                )
            )

    def set_permutation(self):
        """Resets the data permutation based on shuffling settings."""
        if self.shuffle_data.value:
            self.permutation = self._rng.permutation(len(self.data_set))
        else:
            self.permutation = np.arange(len(self.data_set))

    def setup_iteration(self):
        """Calculates index splits for different evaluation phases (Train, Val, Test)."""
        self._batch_progress = 0
        self.set_permutation()
        n_samples = len(self.permutation)
        r_train, r_val, _ = self.splitting_ratio

        train_end = int(r_train * n_samples)
        val_end = train_end + int(r_val * n_samples)

        self._permutation_splits = {
            EvaluationPhase.TRAIN: self.permutation[0:train_end],
            EvaluationPhase.VALIDATION: self.permutation[train_end:val_end],
            EvaluationPhase.TEST: self.permutation[val_end:n_samples],
            EvaluationPhase.ALWAYS: self.permutation[0:n_samples],
        }

    def __len__(self):
        return math.ceil(len(self.data_set) / self.batch_size)

    @property
    def input_ports(self) -> list[InputPort]:
        return []

    @property
    def output_ports(self) -> list[OutputPort]:
        return self._output_ports

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self._batch_size, self.shuffle_data]

    def set_mode(self, new_mode):
        if new_mode != self.mode:
            self._batch_progress = 0  # reset batch
        self.mode = new_mode

    def forward(self):
        """Executes the data loading for one batch.

        This method handles split indexing, batch slicing, and moving data to
        the appropriate device.
        """
        split_indices = self._permutation_splits[self.mode]
        n_split = len(split_indices)

        if n_split == 0:
            return

        bs = self.batch_size

        # Reset progress if we've exhausted the current split
        if self._batch_progress >= n_split:
            self._batch_progress = 0
            if self.shuffle_data.value:
                self._rng.shuffle(split_indices)

        indices = split_indices[self._batch_progress : self._batch_progress + bs]
        batch_data = self.data_set.get_batch(indices)

        # Move batch to device if backend is specified
        if self._device is not None and issubclass(self.backend, DeepLearningBackend):
            batch_data = [self.backend.to(b, self._device) for b in batch_data]

        self._batch_progress += bs
        return tuple(batch_data) if len(batch_data) > 1 else batch_data[0]

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

    def provides_data_in_phase(self, phase: EvaluationPhase) -> bool:
        if phase == EvaluationPhase.TRAIN:
            return self.splitting_ratio[0] > 0.0
        if phase == EvaluationPhase.VALIDATION:
            return self.splitting_ratio[1] > 0.0
        if phase == EvaluationPhase.TEST:
            return self.splitting_ratio[2] > 0.0
        if phase == EvaluationPhase.ALWAYS:
            return (
                self.splitting_ratio[0] > 0.0
                and self.splitting_ratio[1] > 0.0
                and self.splitting_ratio[2] > 0.0
            )
        return False
