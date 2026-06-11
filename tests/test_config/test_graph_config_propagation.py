import pytest
from typing import Annotated, Any
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import (
    BatchAxes,
    FeatureAxes,
    EllipsisAxes,
    AxesDim,
    EllipsisDim,
    GeometryAxes,
)
from qewton.config.errors import DataConfigMismatchError


class MockNode(Node):
    """
    A mock node that allows specifying input/output data configurations
    for testing configuration propagation.
    """

    def __init__(self, name, in_config=None, out_config=None):
        # Store configs so the lambda in forward can access them during _build_ports
        self._in_config = in_config or DataConfiguration.empty()
        self._out_config = out_config or DataConfiguration.empty()
        super().__init__(name=name)

    def forward(
        self, x: Annotated[Any, lambda self: self._in_config]
    ) -> Annotated[Any, lambda self: self._out_config]:
        return x


class TestGraphConfigPropagation:

    def test_linear_propagation(self):
        """
        Tests that a feature dimension propagates backwards through an ellipsis.
        Structure: n1(Batch, ...) -> n2(..., Feature(10))
        Expected: Both ports become (Batch, Feature(10))
        """
        g = Graph()

        # Node 1: output is (Batch, ...)
        n1 = MockNode(
            "n1", out_config=DataConfiguration(BatchAxes(AxesDim(None)), EllipsisAxes())
        )
        # Node 2: input is (..., Feature(10))
        n2 = MockNode(
            "n2",
            in_config=DataConfiguration(
                EllipsisAxes(), FeatureAxes(shape=(AxesDim(10),))
            ),
        )

        # Connect triggers update_data_configurations
        g.connect(n1.output_ports[0], n2.input_ports[0])

        # Both should now have (Batch, Feature(10))
        for node, port in [(n1, n1.output_ports[0]), (n2, n2.input_ports[0])]:
            config = g.dynamic_data_configs[node][port]
            assert len(config.axes) == 3
            assert isinstance(config.axes[0], BatchAxes)
            assert isinstance(config.axes[1], EllipsisAxes)
            assert isinstance(config.axes[-1], FeatureAxes)
            assert config.axes[-1].shape[0].size == 10

    def test_multi_step_propagation(self):
        """
        Tests propagation through multiple nodes.
        Structure: n1(Batch, ...) -> n2(...) -> n3(Batch, Feature(5))
        Expected: Configuration from n3 reaches n1 through n2.
        """
        g = Graph()

        n1 = MockNode(
            "n1", out_config=DataConfiguration(BatchAxes(AxesDim(None)), EllipsisAxes())
        )
        # the ellipsis should point to the same object to view them as identical
        middle_ellipsis = EllipsisAxes()
        n2 = MockNode(
            "n2",
            in_config=DataConfiguration(middle_ellipsis),
            out_config=DataConfiguration(middle_ellipsis),
        )
        n3 = MockNode(
            "n3",
            in_config=DataConfiguration(
                BatchAxes(AxesDim(None)), FeatureAxes(shape=(AxesDim(5),))
            ),
        )

        # Sequence of connections
        g.connect(n1.output_ports[0], n2.input_ports[0])
        g.connect(n2.output_ports[0], n3.input_ports[0])

        # Configuration from n3 should have reached n1
        n1_out = g.dynamic_data_configs[n1][n1.output_ports[0]]
        assert len(n1_out.axes) == 2
        assert isinstance(n1_out.axes[0], BatchAxes)
        assert isinstance(n1_out.axes[1], FeatureAxes)
        assert n1_out.axes[1].shape[0].size == 5

    def test_shape_refinement_with_none(self):
        """
        Tests that partial shape info (None) is refined to concrete values.
        """
        g = Graph()

        # n1: (Batch(None), Feature(None))
        n1 = MockNode(
            "n1",
            out_config=DataConfiguration(
                BatchAxes(AxesDim(None)), FeatureAxes(shape=(AxesDim(None),))
            ),
        )
        # n2: (Batch(32), Feature(128))
        n2 = MockNode(
            "n2",
            in_config=DataConfiguration(
                BatchAxes(AxesDim(32)), FeatureAxes(shape=(AxesDim(128),))
            ),
        )

        g.connect(n1.output_ports[0], n2.input_ports[0])

        n1_out = g.dynamic_data_configs[n1][n1.output_ports[0]]
        assert n1_out.axes[0].shape[0].size == 32
        assert n1_out.axes[1].shape[0].size == 128

    def test_ellipsis_resolution_complex(self):
        """
        Tests complex ellipsis resolution:
        (Batch, ...) connected to (..., Spatial_H, Spatial_W, Feature)
        -> (Batch, Spatial_H, Spatial_W, Feature)
        """
        g = Graph()

        n1 = MockNode(
            "n1", out_config=DataConfiguration(BatchAxes(AxesDim(None)), EllipsisAxes())
        )

        # Spatial dimensions (28x28)
        spatial = GeometryAxes(shape=(AxesDim(28), AxesDim(28)))
        n2 = MockNode(
            "n2",
            in_config=DataConfiguration(
                EllipsisAxes(), spatial, FeatureAxes(shape=(AxesDim(1),))
            ),
        )

        g.connect(n1.output_ports[0], n2.input_ports[0])

        config = g.dynamic_data_configs[n1][n1.output_ports[0]]
        # Expected axes: BatchAxes, GeometryAxes, FeatureAxes
        assert len(config.axes) == 4
        assert isinstance(config.axes[0], BatchAxes)
        assert isinstance(config.axes[1], EllipsisAxes)
        assert isinstance(config.axes[2], GeometryAxes)
        assert isinstance(config.axes[3], FeatureAxes)

        # Verify sizes
        assert [dim.size for dim in config.axes[2].shape] == [28, 28]
        assert config.axes[3].shape[0].size == 1

    def test_ellipsis_removal(self):
        """
        Tests complex ellipsis resolution:
        (Batch, ...) connected to (..., Spatial_H, Spatial_W, Feature)
        -> (Batch, Spatial_H, Spatial_W, Feature)
        """
        g = Graph()

        n1 = MockNode(
            "n1",
            out_config=DataConfiguration(
                BatchAxes(AxesDim(None)),
                GeometryAxes(shape=(AxesDim(28), AxesDim(28))),
                EllipsisAxes(),
            ),
        )

        # Spatial dimensions (28x28)
        spatial = GeometryAxes(shape=(AxesDim(28), AxesDim(28)))
        n2 = MockNode(
            "n2",
            in_config=DataConfiguration(
                EllipsisAxes(), spatial, FeatureAxes(shape=(AxesDim(1),))
            ),
        )

        g.connect(n1.output_ports[0], n2.input_ports[0])

        config = g.dynamic_data_configs[n1][n1.output_ports[0]]
        # Expected axes: BatchAxes, GeometryAxes, FeatureAxes
        assert len(config.axes) == 3
        assert isinstance(config.axes[0], BatchAxes)
        assert isinstance(config.axes[1], GeometryAxes)
        assert isinstance(config.axes[2], FeatureAxes)

        # Verify sizes
        assert [dim.size for dim in config.axes[1].shape] == [28, 28]
        assert config.axes[2].shape[0].size == 1

    def test_diamond_propagation(self):
        """
        Tests propagation in a diamond graph where constraints from the end
        propagate back to the source through multiple paths.
        """
        g = Graph()

        a = MockNode(
            "A", out_config=DataConfiguration(BatchAxes(AxesDim(None)), EllipsisAxes())
        )
        b_middle_ellipsis = EllipsisAxes()
        b = MockNode(
            "B",
            in_config=DataConfiguration(b_middle_ellipsis),
            out_config=DataConfiguration(b_middle_ellipsis),
        )
        c_middle_ellipsis = EllipsisAxes()
        c = MockNode(
            "C",
            in_config=DataConfiguration(c_middle_ellipsis),
            out_config=DataConfiguration(c_middle_ellipsis),
        )

        # D defines the feature size
        d = MockNode(
            "D",
            in_config=DataConfiguration(
                EllipsisAxes(), FeatureAxes(shape=(AxesDim(64),))
            ),
        )

        g.connect(a.output_ports[0], b.input_ports[0])
        g.connect(a.output_ports[0], c.input_ports[0])
        g.connect(b.output_ports[0], d.input_ports[0])

        # Constraint from D (64) should have reached A via B
        config_a = g.dynamic_data_configs[a][a.output_ports[0]]
        config_c_out = g.dynamic_data_configs[c][c.output_ports[0]]

        assert config_a.axes[2].shape[0].size == 64
        assert config_c_out.axes[2].shape[0].size == 64

    def test_propagation_through_graph_node(self):
        """
        Tests data config propagation through a GraphNode.
        Main Graph: n_start (Batch, ...) -> graph_node (encapsulating inner_n1 (...) -> inner_n2 (Batch, Feature(10))) -> n_end
        Expected: All relevant ports should resolve to (Batch, Feature(10)).
        """
        from qewton.graphs.control_nodes.graph_node import GraphNode

        g_main = Graph()

        # 1. Define the inner graph
        g_inner = Graph()
        inner_n1_ellipse = EllipsisAxes()
        inner_n1 = MockNode(
            "inner_n1",
            in_config=DataConfiguration(inner_n1_ellipse),
            out_config=DataConfiguration(inner_n1_ellipse),
        )
        inner_n2_config = DataConfiguration(
            BatchAxes(AxesDim(None)), FeatureAxes(shape=(AxesDim(10),))
        )
        inner_n2 = MockNode(
            "inner_n2", in_config=inner_n2_config, out_config=inner_n2_config
        )
        g_inner.connect(inner_n1.output_ports[0], inner_n2.input_ports[0])

        # 2. Create the GraphNode encapsulating g_inner
        # The GraphNode's input port will be connected to inner_n1.input_ports[0].
        # The GraphNode's output port will be connected from inner_n2.output_ports[0].
        graph_node = GraphNode(
            graph=g_inner,
            input_ports=[inner_n1.input_ports[0]],
            output_ports=[inner_n2.output_ports[0]],
            name="graph_node",
        )

        # 3. Main graph setup
        n_start = MockNode(
            "n_start",
            out_config=DataConfiguration(BatchAxes(AxesDim(None)), EllipsisAxes()),
        )
        n_end = MockNode(
            "n_end",
            in_config=DataConfiguration(
                BatchAxes(AxesDim(None)), FeatureAxes(shape=(AxesDim(10),))
            ),
        )

        # Connect n_start to graph_node's input
        g_main.connect(n_start.output_ports[0], graph_node.input_ports[0])
        # Connect graph_node's output to n_end
        g_main.connect(graph_node.output_ports[0], n_end.input_ports[0])

        # 4. Assertions for main graph ports
        # n_start output should be (Batch, Feature(10))
        n_start_out_config = g_main.dynamic_data_configs[n_start][n_start.output_ports[0]]

        assert len(n_start_out_config.axes) == 2
        assert isinstance(n_start_out_config.axes[0], BatchAxes)
        assert isinstance(n_start_out_config.axes[1], FeatureAxes)
        assert n_start_out_config.axes[1].shape[0].size == 10

        # graph_node input should be (Batch, Feature(10))
        gn_in_config = g_main.dynamic_data_configs[graph_node][graph_node.input_ports[0]]
        assert len(gn_in_config.axes) == 2
        assert isinstance(gn_in_config.axes[0], BatchAxes)
        assert isinstance(gn_in_config.axes[1], FeatureAxes)
        assert gn_in_config.axes[1].shape[0].size == 10

        # graph_node output should be (Batch, Feature(10))
        gn_out_config = g_main.dynamic_data_configs[graph_node][
            graph_node.output_ports[0]
        ]
        assert len(gn_out_config.axes) == 2
        assert isinstance(gn_out_config.axes[0], BatchAxes)
        assert isinstance(gn_out_config.axes[1], FeatureAxes)
        assert gn_out_config.axes[1].shape[0].size == 10

        # n_end input should be (Batch, Feature(10))
        n_end_in_config = g_main.dynamic_data_configs[n_end][n_end.input_ports[0]]
        assert len(n_end_in_config.axes) == 2
        assert isinstance(n_end_in_config.axes[0], BatchAxes)
        assert isinstance(n_end_in_config.axes[1], FeatureAxes)
        assert n_end_in_config.axes[1].shape[0].size == 10

        # 5. Assertions for internal nodes of graph_node
        # inner_n1 output should be (Batch, Feature(10))
        inner_n1_out_config = g_inner.dynamic_data_configs[inner_n1][
            inner_n1.output_ports[0]
        ]
        assert len(inner_n1_out_config.axes) == 2
        assert isinstance(inner_n1_out_config.axes[0], BatchAxes)
        assert isinstance(inner_n1_out_config.axes[1], FeatureAxes)
        assert inner_n1_out_config.axes[1].shape[0].size == 10

        # inner_n2 input should be (Batch, Feature(10))
        inner_n2_in_config = g_inner.dynamic_data_configs[inner_n2][
            inner_n2.input_ports[0]
        ]
        assert len(inner_n2_in_config.axes) == 2
        assert isinstance(inner_n2_in_config.axes[0], BatchAxes)
        assert isinstance(inner_n2_in_config.axes[1], FeatureAxes)
        assert inner_n2_in_config.axes[1].shape[0].size == 10
