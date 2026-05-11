import pytest
from typing import Annotated, Any
from pioneer.graphs.graphs import Graph
from pioneer.graphs.nodes import Node
from pioneer.config.data_configurations import DataConfiguration
from pioneer.config.axes import (
    BatchAxes,
    FeatureAxes,
    EllipsisAxes,
    AxesDim,
    GeometryAxes,
)
from pioneer.config.errors import DataConfigMismatchError


class MockNode(Node):
    """
    A mock node that allows specifying input/output data configurations
    for testing configuration propagation.
    """

    def __init__(self, name, in_config=None, out_config=None):
        # Store configs so the lambda in forward can access them during _build_ports
        self._in_config = in_config or DataConfiguration()
        self._out_config = out_config or DataConfiguration()
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
            assert len(config.axes) == 2
            assert isinstance(config.axes[0], BatchAxes)
            assert isinstance(config.axes[1], FeatureAxes)
            assert config.axes[1].shape[0].size == 10

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
        n2 = MockNode(
            "n2",
            in_config=DataConfiguration(EllipsisAxes()),
            out_config=DataConfiguration(EllipsisAxes()),
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
        b = MockNode(
            "B",
            in_config=DataConfiguration(EllipsisAxes()),
            out_config=DataConfiguration(EllipsisAxes()),
        )
        c = MockNode(
            "C",
            in_config=DataConfiguration(EllipsisAxes()),
            out_config=DataConfiguration(EllipsisAxes()),
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
        assert config_a.axes[1].shape[0].size == 64
