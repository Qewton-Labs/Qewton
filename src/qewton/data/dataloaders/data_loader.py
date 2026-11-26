"""The DataLoader: batches, shuffles and splits a DataSet for the graph."""

from typing import Any
import math
import numpy as np

from qewton.backends import Backend, DEFAULT_DL_BACKEND, DeepLearningBackend, TensorType
from qewton.config.axes import BatchAxes, AxesDim
from qewton.config.data_configurations import DataConfiguration
from qewton.data.dataloaders.base import DataNode
from qewton.data.datasets import DataSet
from qewton.graphs.nodes import NodeState, OutputPort, InputPort
from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
)
from qewton.optim.parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
)


class DataLoader(DataNode[TensorType]):
    """Standard DataLoader module for batching, shuffling, and splitting datasets.

    This node acts as a source in the computation graph, providing batches of
    data to connected algorithms.

    Args:
        data_set (DataSet): The source dataset, split into Train/Validation/
            Test according to `splitting_ratio`.
        batch_size (int | DiscreteHyperparameter | CategoricalHyperparameter):
            Number of samples per batch in training, and in every phase
            without a batch size of its own.
        splitting_ratio (tuple[float, float, float], optional): Proportions
            for (Train, Validation, Test) splits of `data_set`. Ignored for
            the Test split when `test_data_set` is given. Defaults to
            (1.0, 0.0, 0.0).
        shuffle_data (bool | CategoricalHyperparameter, optional): Whether
            to shuffle the indices at the start of an epoch.. Defaults to True.
        shuffle_seed (int | None, optional): Random seed for reproducibility.
            Defaults to None.
        backend (type[Backend] | None, optional): The backend used for data
            types and device transfers.. Defaults to DEFAULT_DL_BACKEND.
        name (str, optional): Name of this data loader. Defaults to "DataLoader".
        test_data_set (DataSet | None, optional): A separate dataset used
            for the Test split instead of the portion of `data_set` that
            `splitting_ratio` would otherwise reserve for it. Must provide
            the same variables as `data_set`. Defaults to None.
        validation_batch_size (int | DiscreteHyperparameter |
            CategoricalHyperparameter | None, optional): Number of samples
            per batch in the Validation split. Defaults to None, i.e.
            `batch_size`.
        test_batch_size (int | DiscreteHyperparameter |
            CategoricalHyperparameter | None, optional): Number of samples
            per batch in the Test split. Defaults to None, i.e. `batch_size`.
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
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
        name: str = "DataLoader",
        test_data_set: DataSet | None = None,
        validation_batch_size: (
            int | DiscreteHyperparameter | CategoricalHyperparameter | None
        ) = None,
        test_batch_size: (
            int | DiscreteHyperparameter | CategoricalHyperparameter | None
        ) = None,
    ):
        self.data_set = data_set
        self.test_data_set = test_data_set
        self.splitting_ratio = splitting_ratio
        self.shuffle_data = HyperParameter.from_value(shuffle_data, "shuffle_data")
        self.shuffle_seed = shuffle_seed
        self._rng = np.random.default_rng(self.shuffle_seed)
        self.permutation = []
        self.test_permutation = []
        self._permutation_splits = {}
        self._phase_data_sets = {}
        if self.data_set.data_set_loaded:
            self._setup_iteration()

        super().__init__(
            batch_size=batch_size,
            name=name,
            state=NodeState.UNINITIALIZED,
            backend=backend,
            validation_batch_size=validation_batch_size,
            test_batch_size=test_batch_size,
        )

        batch_sizes_of_data_set = [self.batch_size, self.validation_batch_size]
        if self.test_data_set is not None:
            assert (
                len(self.test_data_set) >= self.test_batch_size
            ), "Batch can not be larger than the test dataset size."
            assert [c.variable_name for c in self.test_data_set.data_configs] == [
                c.variable_name for c in self.data_set.data_configs
            ], "test_data_set must provide the same variables as data_set."
        else:
            batch_sizes_of_data_set.append(self.test_batch_size)
        largest_batch_size = max(batch_sizes_of_data_set)

        # Build output ports based on dataset configurations. All outputs
        # share one batch axes, bound to the batch size of the current phase.
        self._output_ports = []
        self.batch_axes = BatchAxes(AxesDim(self._phase_batch_size))
        for config in self.data_set.data_configs:
            axes = list(config.axes)
            assert isinstance(axes[0], BatchAxes), "In DataSets, \
                the first axes should be the batch axes."
            assert len(axes[0].shape) == 1, "Multi-dimensional \
                batch axes not supported for batching."
            assert (
                axes[0].shape[0].size >= largest_batch_size
            ), "Batch can not be larger than dataset size."
            axes[0] = self.batch_axes
            new_config = DataConfiguration(
                *axes, dtype=backend.default_dtype if backend else Any
            )
            self._output_ports.append(
                OutputPort(
                    new_config,
                    self,
                    name=config.variable_name,
                )
            )

    def _set_permutation(self):
        """Resets the data permutation based on shuffling settings."""
        if self.shuffle_data.value:
            self.permutation = self._rng.permutation(len(self.data_set))
            if self.test_data_set is not None:
                self.test_permutation = self._rng.permutation(len(self.test_data_set))
        else:
            self.permutation = np.arange(len(self.data_set))
            if self.test_data_set is not None:
                self.test_permutation = np.arange(len(self.test_data_set))

    def _setup_iteration(self):
        """Calculates index splits for different evaluation phases
        (Train, Val, Test).
        """
        self._batch_progress = 0
        self._set_permutation()
        n_samples = len(self.permutation)
        r_train, r_val, _ = self.splitting_ratio

        train_end = int(r_train * n_samples)
        val_end = train_end + int(r_val * n_samples)

        if self.test_data_set is not None:
            test_indices = self.test_permutation
            test_data_set = self.test_data_set
        else:
            test_indices = self.permutation[val_end:n_samples]
            test_data_set = self.data_set

        self._permutation_splits = {
            EvaluationPhase.TRAIN: self.permutation[0:train_end],
            EvaluationPhase.VALIDATION: self.permutation[train_end:val_end],
            EvaluationPhase.TEST: test_indices,
            EvaluationPhase.ALWAYS: self.permutation[0:n_samples],
        }
        self._phase_data_sets = {
            EvaluationPhase.TRAIN: self.data_set,
            EvaluationPhase.VALIDATION: self.data_set,
            EvaluationPhase.TEST: test_data_set,
            EvaluationPhase.ALWAYS: self.data_set,
        }

    def __len__(self):
        return math.ceil(len(self.data_set) / self.batch_size)

    @property
    def training_batches(self) -> int:
        return math.ceil(len(self.data_set) * self.splitting_ratio[0] / self.batch_size)

    @property
    def input_ports(self) -> list[InputPort]:
        return []

    @property
    def output_ports(self) -> list[OutputPort]:
        return self._output_ports

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [*super().hyperparameters, self.shuffle_data]

    def set_mode(self, new_mode):
        if new_mode != self.mode:
            self._batch_progress = 0  # reset batch
        self.mode = new_mode

    def build(self) -> None:
        if not self.data_set.data_set_loaded:
            self.data_set.setup()
            self._setup_iteration()

    def forward(self):
        """Executes the data loading for one batch.

        This method handles split indexing, batch slicing, and moving data to
        the appropriate device.
        """
        split_indices = self._permutation_splits[self.mode]
        n_split = len(split_indices)

        if n_split == 0:
            return

        bs = self.current_batch_size
        # Reset progress if we've exhausted the current split
        if self._batch_progress >= n_split:
            self._batch_progress = 0
            if self.shuffle_data.value:
                self._rng.shuffle(split_indices)

        indices = split_indices[self._batch_progress : self._batch_progress + bs]
        batch_data = self._phase_data_sets[self.mode].get_batch(indices)

        # Move batch to device if backend is specified
        if self._device is not None and issubclass(self.backend, DeepLearningBackend):
            batch_data = [self.backend.to(b, self._device) for b in batch_data]

        self._batch_progress += bs
        return tuple(batch_data) if len(batch_data) > 1 else batch_data[0]

    def cache(self, n_batches=-1):
        return

    def provides_data_in_phase(self, phase: EvaluationPhase) -> bool:
        if phase == EvaluationPhase.TRAIN:
            return self.splitting_ratio[0] > 0.0
        if phase == EvaluationPhase.VALIDATION:
            return self.splitting_ratio[1] > 0.0
        if phase == EvaluationPhase.TEST:
            return self.test_data_set is not None or self.splitting_ratio[2] > 0.0
        if phase == EvaluationPhase.ALWAYS:
            return (
                self.splitting_ratio[0] > 0.0
                and self.splitting_ratio[1] > 0.0
                and (self.test_data_set is not None or self.splitting_ratio[2] > 0.0)
            )
        return False
