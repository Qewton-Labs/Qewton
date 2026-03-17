from .base import Pipeline
from ..algorithms.base import AlgorithmNode, AlgorithmAttributes
from ..constraints.metric_constraint import MetricConstraint, MSEConstraint
from ..data.datasets.base import DataSet
from ..nodes.operations.slice_nodes import SliceNode
from ..nodes.operations.normalization import NormalizationNode, InverseNormalizationNode
from ..optim.hyperparameter.categorical_hyperparameter import BooleanHyperparameter


class MSEDataPipeline(Pipeline):
    """A pipeline that implements a data fitting pipeline."""

    def __init__(
        self,
        dataset: DataSet,
        algorithm: AlgorithmNode,
        constraint: MetricConstraint | None = None,
        apply_normalization: bool = True,
        name="MSEDataPipeline",
    ):
        """
        Args:
            dataset (DataSet): The dataset, providing the data for the training.
            algorithm (AlgorithmNode): The algorithm that should be tested or trained.
            constraint (MetricConstraint): The constraint that should be
                applied/fulfilled.
            apply_normalization (bool, optional): If the input and output of the
                algorithm should be normalized. If algorithm.attributes
                includes AlgorithmAttributes.NORMALIZES_DATA this
                is set to false automatically. Defaults to True.
            name (str, optional): Name of the pipeline. Defaults to "MSEDataPipeline".
        """
        super().__init__(name)
        apply_normalization = (
            apply_normalization
            and not AlgorithmAttributes.NORMALIZES_DATA in algorithm.attributes
        )
        # Build the nodes of the pipeline:
        algo_config_in = algorithm.input_ports[0].data_configuration
        slice_node_input = SliceNode(
            dataset.data_config,
            algo_config_in.feature_axis.variables,  # type: ignore
            name="SliceInput",
        )
        algo_config_out = algorithm.input_ports[0].data_configuration
        slice_node_output = SliceNode(
            dataset.data_config,
            algo_config_out.feature_axis.variables,  # type: ignore
            name="SliceOutput",
        )
        if constraint is None:
            self.mse_constraint = MSEConstraint(algo_config_out)
        else:
            self.mse_constraint = constraint

        if apply_normalization:
            normalize_param = BooleanHyperparameter(True)
            self.normalize_node = NormalizationNode(
                data_config=dataset.data_config[algorithm.input_variable],
                dataset_node=dataset,
                active=normalize_param,
            )
            self.invert_normalize_node = InverseNormalizationNode(
                data_config=algorithm[algorithm.OutputKeys.OUTPUT].data_configuration,
                dataset_node=dataset,
                active=normalize_param,
                name="InverseNormalization",
            )

        # Connect everything
        self.connect(dataset, slice_node_input)
        self.connect(dataset, slice_node_output)
        if apply_normalization:
            self.connect(slice_node_input, self.normalize_node)
            self.connect(self.normalize_node, algorithm)
            self.connect(algorithm, self.invert_normalize_node)
            self.connect(
                self.invert_normalize_node,
                self.mse_constraint[self.mse_constraint.InputKeys.INPUT1],
            )
        else:
            self.connect(slice_node_input, algorithm)
            self.connect(
                algorithm,
                self.mse_constraint[self.mse_constraint.InputKeys.INPUT1],
            )
        self.connect(
            slice_node_output, self.mse_constraint[self.mse_constraint.InputKeys.INPUT2]
        )
        self.validate()
