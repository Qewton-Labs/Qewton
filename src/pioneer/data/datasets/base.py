from typing import Any, Callable
from abc import abstractmethod

from ...config import DataConfiguration
from ...optim.hyperparameter.base import HyperParameter
from ...optim.hyperparameter.number_hyperparameter import (
    DiscreteHyperparameter,
)
from ...optim.hyperparameter.categorical_hyperparameter import (
    CategoricalHyperparameter,
)
from ...config.variables import Variable
from ...config.axis import SpatialAxis, BatchAxis, FeatureAxis
from ...nodes.base import Node, OutputPort, InputPort

# TODO: For now just a simple dataset where the data is provided
# How do we handle splitting the data for training, testing, validation?
# Currently everything is done here, but maybe split this further?
# With the current way we can use the same pipeline for multiple things,
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


class DataSet(Node):

    def __init__(
        self,
        data_config: DataConfiguration,
        data,
        batch_size: int | DiscreteHyperparameter | CategoricalHyperparameter,
        splitting_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        shuffle_data: bool = True,
        name: str = "DataSet",
    ):
        super().__init__(name=name)
        self.data = data
        assert (
            sum(splitting_ratio) <= 1
        ), "Sum of ratio for data splitting should not be greater 1!"
        # TODO: Can we do here any test to check, if the data fits the configuration.
        # But depends on the type of data (numpy, pandas, etc.)...
        self.data_config = data_config
        self.batch_size: HyperParameter = HyperParameter.from_value(
            batch_size, name="Batch Size"
        )
        self.shuffle_data = shuffle_data
        # TODO: Can the ratios be Hyperparameters, we could allow for a
        # CategorialHyperparameter consisting of lists with 3 values?
        self.splitting_ratio = splitting_ratio
        self._batch_progress: int = (
            0  # the last element returned when "run" was called last.
        )

        self._mean = None
        self._std = None
        self.std_eps = 1.0e-5  # small tolerance to add when std is equal 0
        self.out_port = OutputPort(self.data_config, self)

    @classmethod
    def from_data(
        cls,
        data,
        variable: Variable,
        batch_size: int | DiscreteHyperparameter | CategoricalHyperparameter,
        splitting_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        shuffle_data: bool = False,
        name: str = "DataSetNode",
    ):
        # registry decides which subclass to call
        for cond, dataset_cls in DATASET_REGISTRY:
            if cond(data):
                dataset_cls: DataSet = dataset_cls
                return (
                    dataset_cls._from_data_internal(  # pylint: disable=protected-access
                        data, variable, batch_size, splitting_ratio, shuffle_data, name
                    )
                )
        return cls._from_data_internal(
            data, variable, batch_size, splitting_ratio, shuffle_data, name
        )

    @classmethod
    def _from_data_internal(
        cls,
        data,
        variable: Variable,
        batch_size: int | DiscreteHyperparameter | CategoricalHyperparameter,
        splitting_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        shuffle_data: bool = False,
        name: str = "DataSetNode",
    ):
        # TODO: Make this method more general!
        # warnings.warn(
        #     "This method does not handle any case yet, and assumes that \
        #         the first axis is the batch and last contains the variables!",
        #     UserWarning,
        # )
        axes = []
        for i, data_size in enumerate(data.shape):
            if i == 0:
                axes.append(BatchAxis())
            elif i == len(data.shape) - 1:
                axes.append(FeatureAxis(size=data_size, variables=variable))
            else:
                axes.append(SpatialAxis(size=data_size, name=f"spatial_{i}"))

        data_config = DataConfiguration(
            dtype=data.dtype, axes=axes, feature_axis=axes[-1]
        )

        return cls(
            data_config=data_config,
            data=data,
            batch_size=batch_size,
            splitting_ratio=splitting_ratio,
            name=name,
            shuffle_data=shuffle_data,
        )

    @property
    def input_ports(self) -> list[InputPort]:
        return []

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self.batch_size]

    def __len__(self) -> int:
        batch_idx = self.data_config.batch_axis_idx
        return self.data.shape[batch_idx]

    def run(self):
        # TODO: Add batching and splitting of data, currently
        # just a dummy to get a working example. Can this be done in the parent or
        # is this backend dependent? See TorchDataSet
        self.out_port.set_value(self.data)

    def set_mode(self, new_mode):
        if new_mode != self.mode:
            self._batch_progress = 0  # reset batch
        self.mode = new_mode

    def to(self, device):
        pass

    def _compute_mean(self):
        pass

    @property
    def mean(self) -> Any:
        if self._mean is None:
            self._compute_mean()
        return self._mean

    def _compute_std(self):
        pass

    @property
    def std(self) -> Any:
        if self._std is None:
            self._compute_std()
        return self._std

    @abstractmethod
    def compute_pca(self, n_components: int, variable: Variable) -> tuple[Any, Any, Any]:
        """Does a principal component analysis (PCA) on the data.

        Args:
            n_components (int): The number of components that should be used/returned
                in the PCA.
            variable (Variable): For which variables from the data the PCA should be
                carried out.

        Returns:
            tuple: A tuple containing the PCA. The tuple has the form (U, S, V) with
                U = left singular vectors, shape (batch_dim, n_components)
                S = singular values, shape (n_components,)
                V = right singular vectors, shape (n_components, feature_dim)
        """
        return (None, None, None)
