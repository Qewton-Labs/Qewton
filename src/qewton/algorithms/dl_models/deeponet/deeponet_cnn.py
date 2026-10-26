from typing import Annotated, Literal

from qewton.algorithms.dl_models.deeponet.base import DeepONet
from qewton.algorithms.dl_models.deeponet.merge_nodes import DefaultMerger
from qewton.algorithms.dl_models.fcn import FCN
from qewton.algorithms.dl_models.convolutions.encoding import ConvolutionalEncoder
from qewton.algorithms.building_blocks.array_operations import ReshapeAtDim, Unsqueeze
from qewton.algorithms.building_blocks.activation_functions import ReLU
from qewton.backends import DEFAULT_DL_BACKEND, DeepLearningBackend, TensorType
from qewton.graphs.nodes import Node, NodeState
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import Graph
from qewton.config.variables import Variable
from qewton.config.devices import Device, cpu
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.config.axes import EllipsisAxes, FeatureAxes, BatchAxes, AxesDim
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class CNNDeepONet(DeepONet[TensorType]):
    """A DeepONet architecture that uses fully connected networks (FCNs)
    for the trunk and a convolutional neural network (CNN) for the branch.
    The merge node can be configured to handle different output strategies,
    allowing for flexibility in how the outputs of the branch and trunk networks
    are combined. This DeepONet expects inputs of shape:
    - Branch input: (batch_size, branch_input_dim, discretization_shape_branch)
    - Trunk input: (batch_size, discretization_shape_trunk, trunk_input_dim)
    where discretization_shape can be any shape.
    The output will have shape:
    - Output: (batch_size, discretization_shape_trunk, output_dim)

    Args:
        trunk_input (int | Variable): The input dimension for
            the trunk network.
        branch_input (int | Variable): The
    """

    def __init__(
        self,
        trunk_input: int | Variable | HyperParameter,
        branch_input: int | Variable | HyperParameter,
        output: int | Variable,
        trunk_hidden_neurons: int | HyperParameter,
        trunk_hidden_layers: int | HyperParameter,
        channels: tuple[int | HyperParameter, ...],
        conv_kernel_size: HyperParameter | tuple[int, ...],
        intermediate_neurons: int | HyperParameter,
        branch_fcn_hidden_neurons: int | HyperParameter = 50,
        branch_fcn_hidden_layers: int | HyperParameter = 1,
        pooling_kernel_size: int | HyperParameter | tuple[int, ...] = 2,
        pooling_type: Literal["average", "maximum"] = "average",
        output_strategy: DefaultMerger.MERGE_TYPES = "split",
        activations: type[Node] | HyperParameter = ReLU,
        name="CNNDeepONet",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ):
        # Data config info:
        self.batch_axes = BatchAxes(AxesDim(None))
        self.trunk_ellipsis_axes = EllipsisAxes()
        self.feature_axes_branch = self._build_feature_axis(branch_input)
        self.feature_axes_trunk = self._build_feature_axis(trunk_input)
        self.output_axes = self._build_feature_axis(output)

        self.intermediate_neurons = HyperParameter.from_value(
            intermediate_neurons, "Intermediate neurons"
        )
        self.activations = HyperParameter.from_value(activations, "Activations")
        self.output_dim = output if isinstance(output, int) else output.dim
        self.output_strategy: DefaultMerger.MERGE_TYPES = output_strategy
        self.current_device: Device = cpu  # Default device; can be changed later

        self.trunk_net = FCN(
            in_neurons=trunk_input,
            hidden_neurons=trunk_hidden_neurons,
            out_neurons=self.intermediate_neurons,
            n_hidden_layers=trunk_hidden_layers,
            activation=self.activations,
            backend=backend,
        )
        self.branch_net = ConvolutionalEncoder(
            in_channels=branch_input,
            channels=channels,
            out_channels=self.intermediate_neurons,
            conv_kernel_size=conv_kernel_size,
            fcn_hidden_neurons=branch_fcn_hidden_neurons,
            fcn_hidden_layers=branch_fcn_hidden_layers,
            input_shape=None,  # Input shape will be inferred from the first input data
            pooling_kernel_size=pooling_kernel_size,
            pooling_type=pooling_type,
            activation=self.activations,
            backend=backend,
        )
        self.trunk_input: tuple[int, ...] | None = None
        merge_node = self._build_merge_and_update_neurons(backend)

        super().__init__(
            branch_net=self.branch_net,
            trunk_net=self.trunk_net,
            merge_node=merge_node,
            name=name,
            backend=backend,
            **kwargs,
        )
        self._graph.setup()
        self.set_state(NodeState.UNINITIALIZED)

        self.branch_port = self.input_ports[0]
        self.trunk_port = self.input_ports[1]

    def to(self, device):
        self.current_device = device
        self.trunk_net.to(device)
        self.branch_net.to(device)
        return super().to(device)

    def _build_feature_axis(self, input_value):
        if isinstance(input_value, Variable):
            return FeatureAxes(variable=input_value)
        elif isinstance(input_value, int):
            return FeatureAxes(shape=(input_value,))
        elif isinstance(input_value, HyperParameter):
            return FeatureAxes(shape=(input_value.value,))
        else:
            raise TypeError(
                "Input value must be an int, Variable, or HyperParameter, "
                f"but got {type(input_value)}"
            )

    def _build_merge_and_update_neurons(self, backend):
        merge_node = DefaultMerger(
            output_dim=self.output_dim,
            intermediate_neurons=self.intermediate_neurons,
            output_strategy=self.output_strategy,
            backend=backend,
        )
        neurons_value = self.intermediate_neurons.value
        trunk_out = merge_node.trunk_out
        branch_out = merge_node.branch_out
        # The branch network now always needs to unsqueeze the output shape to
        # add missing dimensions, depending on the trunk input:
        if self.trunk_input is not None and len(self.trunk_input) > 1:
            merge_graph = Graph()
            branch_unsqueeze_dim = Unsqueeze(dim=-2, backend=backend)
            branch_add_dim = ReshapeAtDim(
                dim=-2,
                new_shape=(1,)
                * (len(self.trunk_input) - 1),  # -1, since channels are also included
                backend=backend,
            )
            merge_graph.connect(branch_unsqueeze_dim, branch_add_dim)
            merge_graph.connect(branch_add_dim, merge_node.input_ports[0])

            merge_node = GraphNode(
                graph=merge_graph,
                input_ports=[
                    branch_unsqueeze_dim.input_ports[0],
                    merge_node.input_ports[1],
                ],
                output_ports=merge_node.output_ports,
                name="Merge",
                backend=backend,
            )

        # Reset the trunk and branch networks to ensure they are properly
        # initialized with the new output dimensions
        self.trunk_net.reset()
        self.trunk_net.out_neurons.set_value(trunk_out)
        self.trunk_net.setup()
        self.branch_net.reset()
        self.branch_net.out_channels.set_value(branch_out)
        self.branch_net.setup()
        self.intermediate_neurons.set_value(neurons_value)

        return merge_node

    def update_data_configs(self, updated_port, config_dict, dynamic_configs):
        ports = super().update_data_configs(updated_port, config_dict, dynamic_configs)
        if len(self._input_ports) == 2 and updated_port == self._input_ports[1]:
            trunk_config_shape = dynamic_configs[updated_port].shape[1:]
            if len(trunk_config_shape) > 0 and all(
                isinstance(dim, int) for dim in trunk_config_shape
            ):
                self.trunk_input = trunk_config_shape
        return ports

    def reset(self):
        self.set_state(NodeState.UNINITIALIZED)
        return super().reset()

    def setup(self) -> None:
        if self.state == NodeState.UNINITIALIZED:
            self.merge_node = self._build_merge_and_update_neurons(self.backend)
            new_graph = Graph()
            new_graph.connect(self.branch_net, self.merge_node.input_ports[0])
            new_graph.connect(self.trunk_net, self.merge_node.input_ports[1])
            self.setup_graph(
                new_graph,
                input_ports=self.branch_net.input_ports + self.trunk_net.input_ports,
                output_ports=self.merge_node.output_ports,
            )
            self.set_state(NodeState.INITIALIZED)
            if self.current_device != cpu:
                self.to(self.current_device)

    def _branch_config(self):
        return DC(
            self.batch_axes,
            self.feature_axes_branch,
            EllipsisAxes(),
            dtype=self.backend.default_dtype,
        )

    def _trunk_config(self):
        return DC(
            self.batch_axes,
            self.trunk_ellipsis_axes,
            self.feature_axes_trunk,
            dtype=self.backend.default_dtype,
        )

    def _output_config(self):
        return DC(
            self.batch_axes,
            self.trunk_ellipsis_axes,
            self.output_axes,
            dtype=self.backend.default_dtype,
        )

    def forward(
        self,
        branch_input: Annotated[TensorType, _branch_config],
        trunk_input: Annotated[TensorType, _trunk_config],
    ) -> Annotated[TensorType, _output_config]:
        if self.trunk_input is None or self.branch_net.input_shape is None:
            self.trunk_input = trunk_input.shape[1:]
            self.branch_net.input_shape = branch_input.shape[1:]
            self.set_state(NodeState.UNINITIALIZED)
            self.setup()
        elif self.state == NodeState.UNINITIALIZED:
            self.setup()
        self.input_ports[0].set_value(branch_input)
        self.input_ports[1].set_value(trunk_input)
        self.run()
        return self.output_ports[0].value  # type: ignore
