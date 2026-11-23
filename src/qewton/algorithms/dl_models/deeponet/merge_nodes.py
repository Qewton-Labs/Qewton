from typing import Annotated, Literal

from qewton.algorithms.building_blocks.math import Inner, Multiply, Sum
from qewton.algorithms.building_blocks.array_operations import (
    Flatten,
    ReshapeAtDim,
    UnflattenLike,
    Unsqueeze,
)
from qewton.algorithms.building_blocks.creation import Identity
from qewton.config.axes import AxesDim, BatchAxes, EllipsisAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.backends import DEFAULT_DL_BACKEND, DeepLearningBackend, TensorType
from qewton.graphs.nodes import NodeState
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import Graph
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.config.variables import Variable


class DefaultMerger(GraphNode[TensorType]):
    """A default merger node that merges the outputs of the trunk and branch networks
    in a DeepONet architecture. The merging is done by multiplying the outputs of
    the trunk and branch networks and then summing over the last dimension.

    The branch output has the shape `(batch, branch_out)`, the trunk output
    `(batch, *points, trunk_out)` with any number of point dims; the result
    has the shape `(batch, *points, output_dim)`.

    Args:
        output_dim (int | Variable | HyperParameter): The dimension of the final
            output of the DeepONet.
            If the output dimension is greater than 1, the merge node will
            be configured following the specified output strategy.
        intermediate_neurons (int | HyperParameter): The number of intermediate neurons
            in the trunk and branch networks.
        output_strategy (Literal["split", "split_branch", "split_trunk"], optional):
            The strategy that is used to merge the branch and trunk net.
            In case of a single output, the merge node will be an Inner product node.
            In case of multiple outputs, the merge node will be a graph that
            reshapes the outputs of the branch and trunk networks and then performs
            an element-wise multiplication followed by a summation. The reshaping
            is done according to the specified output strategy:
            - "split": Both the branch and trunk networks will have
              output_dim * intermediate_neurons neurons in the last layer, which will
              be reshaped to (output_dim, intermediate_neurons) before merging.
            - "split_branch": Only the branch network will have
              output_dim * intermediate_neurons, the trunk net output will be
              multiplied with all branch net outputs.
            - "split_trunk": Only the trunk network will have
              output_dim * intermediate_neurons, the branch net output will be
              multiplied with all trunk net outputs.
            Defaults to "split".
        backend (type[DeepLearningBackend[TensorType]]): The deep learning backend to use
            for the merger node. Defaults to DEFAULT_DL_BACKEND.
    """

    MERGE_TYPES = Literal["split", "split_branch", "split_trunk"]

    def __init__(
        self,
        output_dim: int | Variable | HyperParameter,
        intermediate_neurons: int | HyperParameter,
        output_strategy: MERGE_TYPES = "split",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
        name: str = "DefaultMerger",
        **kwargs,
    ):
        self.output_dim = HyperParameter.from_value(
            output_dim if isinstance(output_dim, (int, HyperParameter)) else output_dim.dim,
            "Output dim",
        )
        self.intermediate_neurons = HyperParameter.from_value(
            intermediate_neurons, "Intermediate neurons"
        )
        self.output_strategy = output_strategy
        self.trunk_out, self.branch_out = self.output_sizes(
            self.intermediate_neurons, self.output_dim, output_strategy
        )
        self.batch_axes = BatchAxes(AxesDim(None))
        self.points = EllipsisAxes()
        merge_graph, in_ports, out_ports = self._build_merge_graph(backend)
        super().__init__(
            graph=merge_graph,
            input_ports=in_ports,
            output_ports=out_ports,
            backend=backend,
            name=name,
            **kwargs,
        )
        self._graph.setup(plan=False)
        self.set_state(NodeState.UNINITIALIZED)

    @staticmethod
    def output_sizes(
        intermediate_neurons: HyperParameter,
        output_dim: HyperParameter,
        output_strategy: MERGE_TYPES,
    ) -> tuple[HyperParameter, HyperParameter]:
        """The output sizes the strategy requires of the trunk and the branch
        network; for a single output, the split size equals the intermediate
        neurons.

        Returns:
            tuple[HyperParameter, HyperParameter]: The output size of the
                trunk and of the branch network.
        """
        split = intermediate_neurons * output_dim
        trunk = split if output_strategy in ("split", "split_trunk") else intermediate_neurons
        branch = split if output_strategy in ("split", "split_branch") else intermediate_neurons
        return trunk, branch

    def _build_merge_graph(self, backend):
        """The merge of a branch sample with every point of a trunk sample:
        the points are flattened into one dim, the branch is broadcast over
        it, and the points are restored at the end."""
        merge_graph, (branch_in, trunk_in), (merged,) = self._build_core(backend)
        branch = Unsqueeze(dim=1, backend=backend)
        trunk = Identity(backend=backend)
        points = Flatten(start_dim=1, end_dim=-2, backend=backend)
        restore = UnflattenLike(start_dim=1, end_dim=-1, axis=1, backend=backend)
        merge_graph.connect(branch.output_ports[0], branch_in)
        merge_graph.connect(trunk.output_ports[0], points.input_ports[0])
        merge_graph.connect(points.output_ports[0], trunk_in)
        merge_graph.connect(merged, restore.input_ports[0])
        merge_graph.connect(trunk.output_ports[0], restore.input_ports[1])
        return (
            merge_graph,
            [branch.input_ports[0], trunk.input_ports[0]],
            restore.output_ports,
        )

    def _build_core(self, backend):
        """The merge of a branch sample `(batch, 1, branch_out)` with the
        flattened points of a trunk sample `(batch, n, trunk_out)`."""
        output_dim = self.output_dim.value
        merge_graph = Graph()
        in_ports = []
        out_ports = []
        if output_dim == 1:
            merge_node = Inner(backend=backend, keepdims=True)
            merge_graph.add_node(merge_node)
            in_ports = merge_node.input_ports
            out_ports = merge_node.output_ports
        else:
            if self.output_strategy == "split":
                # Both trunk and branch network will have
                # output_dim * intermediate_neurons neurons in the last layer.
                # Which we just reshape to (output_dim, intermediate_neurons).
                reshape_branch = ReshapeAtDim(
                    dim=-1,
                    new_shape=(output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )
                reshape_trunk = ReshapeAtDim(
                    dim=-1,
                    new_shape=(output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )
            elif self.output_strategy == "split_branch":
                reshape_branch = ReshapeAtDim(
                    dim=-1,
                    new_shape=(output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )
                reshape_trunk = Unsqueeze(dim=-2, backend=backend)
            else:  # self.output_strategy == "split_trunk"
                reshape_branch = Unsqueeze(dim=-2, backend=backend)
                reshape_trunk = ReshapeAtDim(
                    dim=-1,
                    new_shape=(output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )

            multiply_node = Multiply(backend=backend)
            sum_node = Sum(axis=-1, backend=backend)

            merge_graph.connect(reshape_branch, multiply_node.input_ports[0])
            merge_graph.connect(reshape_trunk, multiply_node.input_ports[1])
            merge_graph.connect(multiply_node, sum_node)
            in_ports = [reshape_branch.input_ports[0], reshape_trunk.input_ports[0]]
            out_ports = sum_node.output_ports

        return merge_graph, in_ports, out_ports

    def branch_config(self) -> DC:
        return DC(
            self.batch_axes,
            FeatureAxes(shape=(AxesDim(self.branch_out),)),
            dtype=self.backend.default_dtype,
        )

    def trunk_config(self) -> DC:
        return DC(
            self.batch_axes,
            self.points,
            FeatureAxes(shape=(AxesDim(self.trunk_out),)),
            dtype=self.backend.default_dtype,
        )

    def output_config(self) -> DC:
        return DC(
            self.batch_axes,
            self.points,
            FeatureAxes(shape=(AxesDim(self.output_dim),)),
            dtype=self.backend.default_dtype,
        )

    def forward(
        self,
        branch: Annotated[TensorType, branch_config],
        trunk: Annotated[TensorType, trunk_config],
    ) -> Annotated[TensorType, output_config]:
        return super().forward(branch, trunk)

    def reset(self):
        self.set_state(NodeState.UNINITIALIZED)
        return super().reset()

    def build(self) -> None:
        merge_graph, in_ports, out_ports = self._build_merge_graph(self.backend)
        self.setup_graph(
            merge_graph,
            input_ports=in_ports,
            output_ports=out_ports,
        )
