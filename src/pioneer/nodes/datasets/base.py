from typing import Any
import warnings

from ...config import DataConfiguration
from ...optim.hyperparameter.base import HyperParameter, ContinuousHyperparameter
from ...config.variables import Variable
from ...config.axis import SpatialAxis, BatchAxis, FeatureAxis
from ...nodes.base import Node, Port

# TODO: For now just a simple dataset where the data is provided
# How do we handle splitting the data for training, testing, validation?
# Currently everything is done here, but maybe split this further?
#
# We need DataSets that can:
# - load data on the fly from a file/source
# - run other methods/software to create data


class DataSet(Node):

    def __init__(
        self,
        data_config: DataConfiguration,
        data,
        batch_size: int | ContinuousHyperparameter,
        splitting_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        name: str = "DataSetNode",
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
        self.splitting_ratio = splitting_ratio
        # TODO: Can the ratios be Hyperparameters, we could allow for a
        # CategorialHyperparameter consisting of lists with 3 values?

    @classmethod
    def from_data(
        cls,
        data,
        variable: Variable,
        batch_size: int | ContinuousHyperparameter,
        splitting_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        name: str = "DataSetNode",
    ):
        # TODO: Make this method more general!
        warnings.warn(
            "This method does not handle any case yet, and assumes that \
                the first axis is the batch and last contains the variables!",
            UserWarning,
        )
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

        return DataSet(
            data_config=data_config,
            data=data,
            batch_size=batch_size,
            splitting_ratio=splitting_ratio,
            name=name,
        )

    @property
    def input_ports(self) -> dict[str, Port]:
        return {}

    @property
    def output_ports(self) -> dict[str, Port]:
        # TODO: Add mean, std and pca to output ports? Or just 
        # pass this dataset to the PCA-net?
        return {self.OutputKeys.OUTPUT: Port(self.data_config, self, "data_port")}

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self.batch_size]

    def __len__(self) -> int:
        batch_idx = self.data_config.batch_axis_idx
        return self.data.shape[batch_idx]

    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        _ = inputs
        # TODO: Add batching and splitting of data, currently
        # just a dummy to get a working example
        return {self.OutputKeys.OUTPUT: self.data}

    def set_mode(self, new_mode):
        self.mode = new_mode

    def to(self, device):
        # TODO: Check here what type we are
        self.data = self.data.to(device)

    ### TODO: Implement the following code, while solving:
    ###     - Do we always add all the information to the output?
    ###          - If yes: We have to automatically compute the configurations
    ###            for stuff like pca, mean... -> User needs to specify over which
    ###            axis these computations are applied?
    ###          - If no: How is the information passed to another node?
    ###     - Check in run if stuff like the PCA is computed, if not -> do ti

    # def compute_pca(self, n_components):
    #     self.n_components = HyperParameter(dtype=int, state=n_components)
    #     self.pca = ...

    # def pca(self, n_components=None):
    #     if self.pca:
    #         if n_components is None or n_components == self.pca.n_components:
    #             return self.pca
    #         elif n_components < self.pca.n_components:
    #             return self.pca[:n_components]

    #     self.compute_pca(n_components)
    #     return self.pca

    # def compute_mean(self):
    #     self.mean = ...

    # def mean(self):
    #     if self.mean is None:
    #         self.compute_mean()
    #     return self.mean

    # def compute_std(self):
    #     self.std = ...

    # def std(self):
    #     if self.std is None:
    #         self.compute_std()
    #     return self.std
