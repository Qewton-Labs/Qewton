"""DeepONet with fully connected branch and trunk networks."""

from qewton.algorithms.dl_models.deeponet.base import DeepONet, feature_axes
from qewton.algorithms.dl_models.deeponet.merge_nodes import DefaultMerger
from qewton.algorithms.dl_models.fcn import FCN
from qewton.algorithms.building_blocks.array_operations import Flatten
from qewton.algorithms.building_blocks.activation_functions import ReLU
from qewton.backends import DEFAULT_DL_BACKEND, ComputingBackend, TensorType
from qewton.graphs.nodes import Node, NodeState
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import Graph
from qewton.config.variables import Variable
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.config.data_configurations import DataConfiguration, DynamicDataConfiguration
from qewton.config.axes import AxesDim, EllipsisAxes, BatchAxes
from qewton.config.dtypes import Number


class FCNDeepONet(DeepONet[TensorType]):
    """DeepONet with fully connected branch and trunk networks.

    Both networks are :class:`~qewton.algorithms.dl_models.fcn.FCN`
    instances; their outputs are combined by a
    :class:`~qewton.algorithms.dl_models.deeponet.merge_nodes.DefaultMerger`.
    All non-batch axes of the branch input are flattened before the branch
    FCN. Expected shapes:

    - branch input: ``(batch_size, *discretization_shape, branch_input_dim)``
    - trunk input: ``(batch_size, *discretization_shape, trunk_input_dim)``
    - output: ``(batch_size, *discretization_shape, output_dim)``

    `discretization_shape` is arbitrary, but must be the same for branch
    and trunk.

    Args:
        trunk_input: The input dimension of the trunk network, or the input
            variable.
        branch_input: The number of features of the branch input at each of
            its points, or the input variable. The branch network takes a
            branch sample flattened; its input size (`branch_size`) is
            inferred from the connected data.
        output: The output dimension of the DeepONet, or the output variable.
        trunk_hidden_neurons: The number of neurons in each hidden layer of the
            trunk network.
        branch_hidden_neurons: The number of neurons in each hidden layer of
            the branch network.
        trunk_hidden_layers: The number of hidden layers of the trunk network.
        branch_hidden_layers: The number of hidden layers of the branch
            network.
        intermediate_neurons: The number of features per output dimension that
            branch and trunk output before merging.
        output_strategy: How branch and trunk outputs are merged for multiple
            outputs:

            - ``"split"``: both networks output
              ``output_dim * intermediate_neurons`` values, reshaped to
              ``(output_dim, intermediate_neurons)`` before merging.
            - ``"split_branch"``: only the branch network outputs
              ``output_dim * intermediate_neurons`` values; the trunk output is
              multiplied with each of them.
            - ``"split_trunk"``: only the trunk network outputs
              ``output_dim * intermediate_neurons`` values; the branch output is
              multiplied with each of them.

            For a single output, the merge is an inner product.
        activations: The activation node class of both networks.
        name (str, optional): The name of the node. Defaults to
            ``"FCNDeepONet"``.
        backend: The backend used for computations.
        **kwargs: Passed on to
            :class:`~qewton.algorithms.dl_models.deeponet.base.DeepONet`.

    Attributes:
        output_dim (int): The output dimension.
        output_strategy (str): The merge strategy.
        trunk_input (int or Variable or HyperParameter): The input dimension
            or variable of the trunk.
        branch_input (int or Variable or HyperParameter): The number of
            features or the variable of the branch input.
        branch_size (HyperParameter): The input size of the branch network,
            inferred from the connected data.
        intermediate_neurons (HyperParameter): Architecture hyperparameter
            shared by trunk and branch.
        activations (HyperParameter): Architecture hyperparameter shared by
            trunk and branch.
        trunk_hidden_neurons (HyperParameter): Architecture hyperparameter of
            the trunk net.
        trunk_n_hidden_layers (HyperParameter): Architecture hyperparameter of
            the trunk net.
        branch_hidden_neurons (HyperParameter): Architecture hyperparameter of
            the branch net.
        branch_n_hidden_layers (HyperParameter): Architecture hyperparameter
            of the branch net.
        branch_port (InputPort): The input port of the branch.
        trunk_port (InputPort): The input port of the trunk.
    """

    def __init__(
        self,
        trunk_input: int | Variable | HyperParameter,
        branch_input: int | Variable | HyperParameter,
        output: int | Variable,
        trunk_hidden_neurons: int | HyperParameter,
        branch_hidden_neurons: int | HyperParameter,
        trunk_hidden_layers: int | HyperParameter,
        branch_hidden_layers: int | HyperParameter,
        intermediate_neurons: int | HyperParameter,
        output_strategy: DefaultMerger.MERGE_TYPES = "split",
        activations: type[Node] | HyperParameter = ReLU,
        name="FCNDeepONet",
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ):
        # The points of a branch sample (e.g. sensors) and of a trunk sample
        # (where the output is evaluated) are independent; the output has
        # the points of the trunk.
        self.branch_points = EllipsisAxes()
        self.trunk_points = EllipsisAxes()
        self.batch_axes = BatchAxes(None)

        self.output_variable = output if isinstance(output, Variable) else None
        self.output_dim = HyperParameter.from_value(
            output if isinstance(output, int) else output.dim, "Output dim"
        )
        self.output_strategy: DefaultMerger.MERGE_TYPES = output_strategy
        self.intermediate_neurons = HyperParameter.from_value(
            intermediate_neurons, "Intermediate neurons"
        )
        self.trunk_input = trunk_input
        self.branch_input = branch_input
        self.branch_size = HyperParameter.inferred("Branch input size")
        self.trunk_hidden_neurons = HyperParameter.from_value(
            trunk_hidden_neurons, "Trunk hidden neurons"
        )
        self.branch_hidden_neurons = HyperParameter.from_value(
            branch_hidden_neurons, "Branch hidden neurons"
        )
        self.trunk_n_hidden_layers = HyperParameter.from_value(
            trunk_hidden_layers, "Trunk hidden layers"
        )
        self.branch_n_hidden_layers = HyperParameter.from_value(
            branch_hidden_layers, "Branch hidden layers"
        )
        self.activations = HyperParameter.from_value(activations, "Activations")

        merge_node, branch_net, trunk_net = self._build_network(backend)

        super().__init__(
            branch_net=branch_net,
            trunk_net=trunk_net,
            merge_node=merge_node,
            name=name,
            backend=backend,
            state=NodeState.UNINITIALIZED,
            **kwargs,
        )
        self.branch_port = self.input_ports[0]
        self.trunk_port = self.input_ports[1]
        # The merge node only knows the output dimension; the output port
        # carries the output Variable, if one was given.
        self.output_ports[0].update_static_data_configuration(self.output_data_config())

    def _build_network(self, backend):
        merge_node = DefaultMerger(
            output_dim=self.output_dim,
            intermediate_neurons=self.intermediate_neurons,
            output_strategy=self.output_strategy,
            backend=backend,
        )

        trunk_net = FCN(
            in_neurons=self.trunk_input,
            hidden_neurons=self.trunk_hidden_neurons,
            out_neurons=merge_node.trunk_out,
            n_hidden_layers=self.trunk_n_hidden_layers,
            activation=self.activations,
            backend=backend,
        )
        branch_flatting = Flatten(start_dim=1, backend=backend)
        branch_fcn = FCN(
            in_neurons=self.branch_size,
            hidden_neurons=self.branch_hidden_neurons,
            out_neurons=merge_node.branch_out,
            n_hidden_layers=self.branch_n_hidden_layers,
            activation=self.activations,
            backend=backend,
        )
        branch_graph = Graph()
        branch_graph.connect(branch_flatting, branch_fcn)
        branch_net = GraphNode(
            graph=branch_graph,
            input_ports=branch_flatting.input_ports,
            output_ports=branch_fcn.output_ports,
            name="BranchNet",
            backend=backend,
        )

        return merge_node, branch_net, trunk_net

    def relate(
        self, branch: DynamicDataConfiguration, trunk: DynamicDataConfiguration
    ) -> None:
        """The branch network takes a branch sample flattened: all its dims
        but the batch dim have `branch_size` values together."""
        branch.flatten(1).merge_dim(-1, AxesDim(self.branch_size))

    def build(self) -> None:
        """Rebuild branch, trunk and merge node for the current
        hyperparameter values and connect them."""
        self.merge_node, self.branch_net, self.trunk_net = self._build_network(
            self.backend
        )

        new_graph = Graph()
        new_graph.connect(self.branch_net, self.merge_node.input_ports[0])
        new_graph.connect(self.trunk_net, self.merge_node.input_ports[1])
        self.setup_graph(
            new_graph,
            input_ports=self.branch_net.input_ports + self.trunk_net.input_ports,
            output_ports=self.merge_node.output_ports,
        )

    def branch_data_config(self):
        """Build the expected branch input data configuration.

        Returns:
            DataConfiguration: A batch axis, the points of a branch sample
                and the branch feature axis.
        """
        return DataConfiguration(
            self.batch_axes,
            self.branch_points,
            feature_axes(self.branch_input),
            dtype=self.backend.default_dtype,
        )

    def trunk_data_config(self):
        """Build the expected trunk input data configuration.

        Returns:
            DataConfiguration: A batch axis, the points of a trunk sample and
                the trunk feature axis.
        """
        return DataConfiguration(
            self.batch_axes,
            self.trunk_points,
            feature_axes(self.trunk_input),
            dtype=self.backend.default_dtype,
        )

    def output_data_config(self):
        """Build the output data configuration.

        Returns:
            DataConfiguration: A batch axis, the points of the trunk sample
                and the output feature axis.
        """
        return DataConfiguration(
            self.batch_axes,
            self.trunk_points,
            feature_axes(self.output_variable or self.output_dim),
            dtype=self.backend.default_dtype,
        )

    def forward(
        self,
        branch_input: Number[TensorType, branch_data_config],
        trunk_input: Number[TensorType, trunk_data_config],
    ) -> Number[TensorType, output_data_config]:
        """Evaluate the DeepONet.

        Args:
            branch_input: The branch input, see :meth:`branch_data_config`.
            trunk_input: The trunk input, see :meth:`trunk_data_config`.

        Returns:
            TensorType: The output, see :meth:`output_data_config`.
        """
        return super().forward(branch_input, trunk_input)
