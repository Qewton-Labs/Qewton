from typing import Annotated

from qewton.graphs.control_nodes.data_processing_node import DataProcessingNode
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.data.dataloaders.base import DataNode
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.algorithms.dl_models.fcn import FCN
from qewton.config.variables import Variable
from qewton.data.data_processing.pca import PCANode, InversePCANode
from qewton.data.data_processing.normalization import (
    StdNormalizationNode,
    InverseStdNormalizationNode,
)
from qewton.optim.base import EvaluationPhase
from qewton.graphs.nodes import NodeState
from qewton.graphs.graphs import Graph
from qewton.config.axes import EllipsisAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration


class PCANet(GraphNode[TensorType], DataProcessingNode[TensorType]):
    """A pre build FCN connected to a PCA. The input is first transformed
    via a PCA on the input data, then feed to the FCN, and finally the output
    is transformed back via an inverse PCA.
    An optional normalization of the input and output data can be applied
    before and after the PCA. The node has two input ports, one for the
    input data and one for the output data, and one output port for
    the final output of the network.

    Args:
        input_variable (Variable): The input variable for the network.
        output_variable (Variable): The output variable for the network.
        pca_n_input (int | HyperParameter): The number of principal components
            to keep for the input PCA.
        pca_n_output (int | HyperParameter): The number of principal components
            to keep for the output PCA.
        data_source_node (DataNode[TensorType]): The data source node providing
            the original data set for PCA fitting. It has to contain both the
            input and output data.
        fcn_hidden_layers (int | HyperParameter): The number of hidden layers
            in the FCN.
        fcn_hidden_neurons (int | HyperParameter): The number of neurons in
            each hidden layer of the FCN.
        normalize_data (bool | HyperParameter, optional): If a normalization
            of the input and output data should be applied before
            the PCA. Defaults to True.
        normalization_eps (float, optional): A small tolerance added to the
            normalization to circumvent division by zero. Defaults to 1.0e-6.
        name (str, optional): Defaults to "PCANet".
        backend (type[ComputingBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        input_variable: Variable,
        output_variable: Variable,
        pca_n_input: int | HyperParameter,
        pca_n_output: int | HyperParameter,
        data_source_node: DataNode[TensorType],
        fcn_hidden_layers: int | HyperParameter,
        fcn_hidden_neurons: int | HyperParameter,
        normalize_data: bool | HyperParameter = True,
        normalization_eps: float = 1.0e-6,
        name: str = "PCANet",
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        pca_n_input = HyperParameter.from_value(pca_n_input, "PCA input n")
        pca_n_output = HyperParameter.from_value(pca_n_output, "PCA output n")
        self.normalize_data = HyperParameter.from_value(
            normalize_data, "Normalization active"
        )

        self.input_variable = input_variable
        self.output_variable = output_variable
        self.data_source_node = data_source_node

        # Inner nodes:
        self.fcn = FCN(
            in_neurons=pca_n_input,
            hidden_neurons=fcn_hidden_neurons,
            out_neurons=pca_n_output,
            n_hidden_layers=fcn_hidden_layers,
            backend=backend,
        )
        self.input_pca = PCANode(
            n=pca_n_input, data_source_node=data_source_node, backend=backend
        )
        self.output_pca = PCANode(
            n=pca_n_output, data_source_node=data_source_node, backend=backend
        )
        self.inverse_pca = InversePCANode(self.output_pca)
        self.normalize_input = StdNormalizationNode(
            data_source_node=data_source_node, eps=normalization_eps, backend=backend
        )
        self.normalize_output = StdNormalizationNode(
            data_source_node=data_source_node, eps=normalization_eps, backend=backend
        )
        self.inverse_normalization = InverseStdNormalizationNode(
            std_node=self.normalize_output
        )
        graph, in_ports, out_port = self._build_network()
        self.ellipsis_axes: EllipsisAxes = EllipsisAxes()
        super().__init__(
            name=name,
            graph=graph,
            input_ports=in_ports,
            output_ports=[out_port],
            backend=backend,
            data_source_node=data_source_node,
        )
        # self._graph.setup()

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return (
            self.fcn.hyperparameters
            + self.input_pca.hyperparameters
            + self.output_pca.hyperparameters
            + self.normalize_input.hyperparameters
            + self.normalize_output.hyperparameters
        )

    def _build_network(self):
        graph = Graph()
        graph.add_node(self.output_pca)
        graph.connect(self.input_pca.output, self.fcn)
        graph.connect(self.fcn, self.inverse_pca)
        if self.normalize_data.current_value:
            graph.connect(self.normalize_input, self.input_pca)
            graph.connect(self.normalize_output, self.output_pca)
            graph.connect(self.inverse_pca, self.inverse_normalization)
            return (
                graph,
                [
                    self.normalize_input.input_ports[0],
                    self.normalize_output.input_ports[0],
                ],
                self.inverse_normalization.output_ports[0],
            )
        return (
            graph,
            [self.input_pca.input, self.output_pca.input],
            self.output_pca.output,
        )

    def reset(self):
        self._state = NodeState.UNINITIALIZED
        self.input_pca.reset()
        self.output_pca.reset()
        self.inverse_pca.reset()
        self.fcn.reset()
        return super().reset()

    def setup(self, graph: Graph):
        # First reset all internal nodes:
        self.input_pca.reset()
        self.output_pca.reset()
        self.inverse_pca.reset()
        self.fcn.reset()
        # Now setup the nodes:
        self.fcn.setup()
        # Collect all data:
        total_data_input = []
        total_data_output = []
        for _ in range(self.data_source_node.training_batches):
            in_edge = graph.run_to(last_node=self, mode=EvaluationPhase.TRAIN)
            total_data_input.append(in_edge[self.input_ports[0]].from_port.value)
            total_data_output.append(in_edge[self.input_ports[1]].from_port.value)
        self.fit(total_data_input, total_data_output)
        # Build the main computation network
        new_graph, in_ports, out_port = self._build_network()
        self.setup_graph(
            new_graph,
            input_ports=in_ports,
            output_ports=[out_port],
        )

    def fit(
        self, data_batch: list[TensorType], data_batch_output: list[TensorType]
    ) -> None:
        # Pass it into the internal nodes:
        if self.normalize_data.current_value:
            self.normalize_input.fit(data_batch)
            self.normalize_output.fit(data_batch_output)
            data_batch = [self.normalize_input(x) for x in data_batch]
            data_batch_output = [self.normalize_output(x) for x in data_batch_output]
        self.input_pca.fit(data_batch)
        self.output_pca.fit(data_batch_output)

    def in_data_config(self):
        return DataConfiguration(
            EllipsisAxes(),
            FeatureAxes(variable=self.input_variable),
            dtype=self.backend.default_dtype,
        )

    def out_data_config(self):
        return DataConfiguration(
            self.ellipsis_axes,
            FeatureAxes(variable=self.output_variable),
            dtype=self.backend.default_dtype,
        )

    def forward(
        self,
        parameter_input: Annotated[TensorType, in_data_config],
        solution_input: Annotated[TensorType | None, out_data_config] = None,
    ) -> Annotated[TensorType, out_data_config]:
        if self.normalize_data.current_value:
            parameter_input = self.normalize_input(parameter_input)
        pca_transformed_input = self.input_pca(parameter_input)[0]
        fcn_out = self.fcn(pca_transformed_input)
        pca_transformed_output = self.inverse_pca(fcn_out)
        if self.normalize_data.current_value:
            pca_transformed_output = self.inverse_normalization(pca_transformed_output)
        return pca_transformed_output
