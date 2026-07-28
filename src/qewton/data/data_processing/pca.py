import math
from typing import Annotated

from qewton.graphs.control_nodes.data_processing_node import DataProcessingNode
from qewton.data.dataloaders.base import DataNode
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.graphs.nodes import NodeState
from qewton.graphs.graphs import Graph
from qewton.config.axes import EllipsisAxes, BatchAxes, AxesDim, FeatureAxes
from qewton.config.data_configurations import DataConfiguration


class PCANode(DataProcessingNode[TensorType]):
    """A node that performs a principal component analysis (PCA) on its
    input data.
    To construct the PCA, in the setup phase all training data is collect
    by (partially) evaluating the graph it belongs to.

    Args:
        n (int | HyperParameter): The number of principal components to
            keep.
        data_source_node (DataNode): The original source of data. Does
            not have to be node which is directly connected to this PCANode, only
            the original data loader providing the original data set.
        scale (bool | HyperParameter, optional): If the input
            data after the representation in the PCA-basis should be
            scaled by the principal components. Defaults to True.
        name (str | None, optional): Defaults to "PCA Node".
        backend (type[ComputingBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        n: int | HyperParameter,
        data_source_node: DataNode,
        scale: bool | HyperParameter = True,
        name: str | None = "PCA Node",
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        self.n = HyperParameter.from_value(n, "PCA Components of" + self.name)
        self.scale = HyperParameter.from_value(
            scale, "Scale by Principal Comp." + self.name
        )
        # Data config. properties:
        self.batch_axes = BatchAxes(AxesDim(None))
        self.flattened_dim = AxesDim(None)
        super().__init__(data_source_node, name, backend)

        self.backend: type[ComputingBackend[TensorType]] = backend
        self.data_source_node: DataNode = data_source_node

        self.pca_u: TensorType
        self.pca_s: TensorType
        self.pca_v: TensorType
        self.original_shape: tuple[int, ...]
        # Ports for the PCA coefficients
        self.input = self.input_ports[0]
        self.output = self.output_ports[0]
        self.port_u = self.output_ports[1]
        self.port_u.name = "U" + self.name
        self.port_s = self.output_ports[2]
        self.port_s.name = "S" + self.name
        self.port_v = self.output_ports[3]
        self.port_v.name = "V" + self.name

    def reset(self):
        self._state = NodeState.UNINITIALIZED
        self._set_port_values(None, None, None)
        return super().reset()

    def setup(self, graph: Graph) -> None:
        if self._state == NodeState.INITIALIZED:
            return
        # First collect all data:
        total_data = []
        for _ in range(self.data_source_node.training_batches):
            in_edge = graph.run_to(last_node=self, mode=EvaluationPhase.TRAIN)
            total_data.append(in_edge[self.input_ports[0]].from_port.value)
        self.fit(total_data)

    def fit(self, data_batch):
        # The batch is assumed to be on axis = 0
        self.original_shape = tuple(data_batch[0].shape)
        total_data = self.backend.math.concatenate(data_batch, axis=0)
        self.flattened_dim.update_size(total_data.shape[-1])
        # Now apply the PCA, for this flatten everything except the first
        # axis:
        total_data = self.backend.math.flatten(total_data, 1)
        self.pca_u, self.pca_s, self.pca_v = self.backend.linalg.pca(
            total_data, q=self.n.current_value
        )
        self._set_port_values(self.pca_u, self.pca_s, self.pca_v)

        self._state = NodeState.INITIALIZED

    def to(self, device):
        if self.state != NodeState.UNINITIALIZED:
            self.pca_u = self.backend.to(self.pca_u, device=device)
            self.pca_s = self.backend.to(self.pca_s, device=device)
            self.pca_v = self.backend.to(self.pca_v, device=device)

            self._set_port_values(self.pca_u, self.pca_s, self.pca_v)
        return super().to(device)

    def _set_port_values(self, pca_u, pca_s, pca_v):
        self.port_u.set_value(pca_u)
        self.port_s.set_value(pca_s)
        self.port_v.set_value(pca_v)

    def x_data_config(self):
        return DataConfiguration(self.batch_axes, EllipsisAxes())

    def out_data_config(self):
        return DataConfiguration(
            self.batch_axes, FeatureAxes(shape=(self.n.current_value,))
        )

    def out_u_config(self):
        return DataConfiguration(
            self.batch_axes, FeatureAxes(shape=(self.n.current_value,))
        )

    def out_s_config(self):
        return DataConfiguration(FeatureAxes(shape=(self.n.current_value,)))

    def out_v_config(self):
        return DataConfiguration(
            self.batch_axes, FeatureAxes(shape=(self.flattened_dim,))
        )

    def forward(self, x: Annotated[TensorType, x_data_config]) -> tuple[
        Annotated[TensorType, out_data_config],
        Annotated[TensorType, out_u_config],
        Annotated[TensorType, out_s_config],
        Annotated[TensorType, out_v_config],
    ]:
        flatt_x = self.backend.math.flatten(x, 1)
        pca_coefficients = self.backend.math.matmul(flatt_x, self.pca_v)
        if self.scale.current_value:
            scaling = self.pca_s * math.sqrt(1 / (len(self.pca_u) - 1))
            pca_coefficients /= scaling
        return pca_coefficients, self.pca_u, self.pca_s, self.pca_v


class InversePCANode(DataProcessingNode[TensorType]):
    """Applies a inverse PCA, given some coefficients it uses the
    PCA-basis to map back to the original one.

    Args:
        pca_node (PCANode[TensorType]): The PCANode that yields the PCA basis
            that is required for the inverse transformation.
        name (str | None, optional): Defaults to "Inverse PCA Node"
    """

    def __init__(
        self,
        pca_node: PCANode[TensorType],
        name: str | None = "Inverse PCA Node",
    ) -> None:
        self.batch_axes = BatchAxes(AxesDim(None))
        super().__init__(pca_node, name, backend=pca_node.backend)
        self.data_source_node: PCANode = pca_node
        self.backend: type[ComputingBackend[TensorType]] = pca_node.backend

    def setup(self, graph: Graph) -> None:
        if self.data_source_node.state == NodeState.UNINITIALIZED:
            raise RuntimeError(
                f"PCA Node {self.data_source_node} has not been setup yet!"
            )
        self._state = NodeState.INITIALIZED

    def x_data_config(self):
        return DataConfiguration(
            self.batch_axes,
            FeatureAxes(shape=(self.data_source_node.n.current_value,)),
        )

    def out_data_config(self):
        # TODO: Here its hard to find the shape, its only known later, once
        # we pass this through.
        # Maybe one could interfere this from the dataloader of the pca node?
        return DataConfiguration(self.batch_axes, EllipsisAxes())

    def forward(
        self, pca_coefficients: Annotated[TensorType, x_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        # Do inverse scaling
        if self.data_source_node.scale.current_value:
            scaling = self.data_source_node.pca_s * math.sqrt(
                1 / (len(self.data_source_node.pca_u) - 1)
            )
            pca_coefficients = scaling * pca_coefficients
        # Go back to original dimension
        flatt_x = self.backend.math.matmul(
            pca_coefficients, self.backend.math.transpose(self.data_source_node.pca_v)
        )
        original_shape = (len(flatt_x),) + self.data_source_node.original_shape[1:]
        return self.backend.math.reshape(flatt_x, shape=original_shape)
