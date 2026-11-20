from typing import Annotated, Literal

from qewton.algorithms.dl_models.deeponet.base import DeepONet, feature_axes
from qewton.algorithms.dl_models.deeponet.merge_nodes import DefaultMerger
from qewton.algorithms.dl_models.fcn import FCN
from qewton.algorithms.dl_models.convolutions.encoding import ConvolutionalEncoder
from qewton.algorithms.building_blocks.activation_functions import ReLU
from qewton.backends import DEFAULT_DL_BACKEND, DeepLearningBackend, TensorType
from qewton.graphs.nodes import Node, NodeState
from qewton.graphs.graphs import Graph
from qewton.config.variables import Variable
from qewton.config.devices import Device, cpu
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.config.axes import EllipsisAxes, BatchAxes, AxesDim, GeometryAxes
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
        trunk_input (int | Variable | HyperParameter): The number of features
            of a trunk point, or the trunk input variable.
        branch_input (int | Variable | HyperParameter): The number of
            channels of the branch input, or the branch input variable.
        branch_shape (tuple[int | HyperParameter, ...] | None): The spatial
            shape of a branch sample (`discretization_shape_branch`); None
            to infer it from the connected data.
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
        branch_shape: tuple[int | HyperParameter, ...] | None = None,
        name="CNNDeepONet",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ):
        # Data config info:
        self.batch_axes = BatchAxes(AxesDim(None))
        self.trunk_ellipsis_axes = EllipsisAxes()
        self.feature_axes_branch = feature_axes(branch_input)
        self.feature_axes_trunk = feature_axes(trunk_input)
        self.output_axes = feature_axes(output)

        self.intermediate_neurons = HyperParameter.from_value(
            intermediate_neurons, "Intermediate neurons"
        )
        self.activations = HyperParameter.from_value(activations, "Activations")
        self.output_dim = HyperParameter.from_value(
            output if isinstance(output, int) else output.dim, "Output dim"
        )
        self.output_strategy: DefaultMerger.MERGE_TYPES = output_strategy
        trunk_out, branch_out = DefaultMerger.output_sizes(
            self.intermediate_neurons, self.output_dim, output_strategy
        )
        self.current_device: Device = cpu  # Default device; can be changed later

        self.trunk_net = FCN(
            in_neurons=trunk_input,
            hidden_neurons=trunk_hidden_neurons,
            out_neurons=trunk_out,
            n_hidden_layers=trunk_hidden_layers,
            activation=self.activations,
            backend=backend,
        )
        self.branch_net = ConvolutionalEncoder(
            in_channels=branch_input,
            channels=channels,
            out_channels=branch_out,
            conv_kernel_size=conv_kernel_size,
            fcn_hidden_neurons=branch_fcn_hidden_neurons,
            fcn_hidden_layers=branch_fcn_hidden_layers,
            input_shape=branch_shape,
            pooling_kernel_size=pooling_kernel_size,
            pooling_type=pooling_type,
            activation=self.activations,
            backend=backend,
        )
        merge_node = self._build_merge(backend)

        super().__init__(
            branch_net=self.branch_net,
            trunk_net=self.trunk_net,
            merge_node=merge_node,
            name=name,
            backend=backend,
            **kwargs,
        )
        self.set_state(NodeState.UNINITIALIZED)

        self.branch_port = self.input_ports[0]
        self.trunk_port = self.input_ports[1]

    def to(self, device):
        self.current_device = device
        self.trunk_net.to(device)
        self.branch_net.to(device)
        return super().to(device)

    def _build_merge(self, backend):
        return DefaultMerger(
            output_dim=self.output_dim,
            intermediate_neurons=self.intermediate_neurons,
            output_strategy=self.output_strategy,
            backend=backend,
        )

    def reset(self):
        self.set_state(NodeState.UNINITIALIZED)
        return super().reset()

    def build(self) -> None:
        # Trunk and branch are built again with the merge node.
        self.trunk_net.reset()
        self.trunk_net.setup()
        self.branch_net.reset()
        self.branch_net.setup()
        self.merge_node = self._build_merge(self.backend)
        new_graph = Graph()
        new_graph.connect(self.branch_net, self.merge_node.input_ports[0])
        new_graph.connect(self.trunk_net, self.merge_node.input_ports[1])
        self.setup_graph(
            new_graph,
            input_ports=self.branch_net.input_ports + self.trunk_net.input_ports,
            output_ports=self.merge_node.output_ports,
        )
        if self.current_device != cpu:
            self.to(self.current_device)

    def _branch_config(self):
        space = GeometryAxes(shape=tuple(AxesDim(s) for s in self.branch_net.input_shape))
        return DC(
            self.batch_axes,
            self.feature_axes_branch,
            space,
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
        return super().forward(branch_input, trunk_input)
