from typing import Annotated

import pytest
import torch

from qewton.algorithms.building_blocks.array_operations import (
    ConcatVariables,
    SplitVariables,
)
from qewton.backends import TensorType, TorchBackend
from qewton.config.axes import FeatureAxes, BatchAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node


def _source(value, config: DataConfiguration, **kwargs):
    class Source(Node[TensorType]):
        def forward(self) -> Annotated[TensorType, config]:
            return value

    return Source(backend=TorchBackend, **kwargs)


class TestSplitVariables:
    def test_output_ports_are_named_and_typed_from_split_into(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        split = SplitVariables([X, Y], backend=TorchBackend)
        assert [p.name for p in split.output_ports] == ["x", "y"]
        # the static config is real from construction, not a wildcard
        # placeholder - see the module docstring for why this matters
        assert split.output_ports[0].data_configuration.variables is X
        assert split.output_ports[1].data_configuration.variables is Y

    def test_get_output_port_accepts_str_or_variable(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        split = SplitVariables([X, Y], backend=TorchBackend)
        assert split.get_output_port("y") is split.output_ports[1]
        assert split.get_output_port(Y) is split.output_ports[1]

    def test_splits_a_real_tensor_along_the_feature_axis(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        data = torch.rand(10, 2)
        config = DataConfiguration(BatchAxes(10), FeatureAxes(X * Y))
        source = _source(data, config, name="Source")
        split = SplitVariables([X, Y], backend=TorchBackend, name="Split")

        graph = Graph()
        graph.add_node(source)
        graph.add_node(split)
        graph.connect(source.output_ports[0], split.input_ports[0])
        graph.setup()
        graph.run()

        assert torch.allclose(split.output_ports[0].value, data[:, :1])
        assert torch.allclose(split.output_ports[1].value, data[:, 1:])

    def test_a_multi_component_piece_can_stay_whole(self):
        """A piece doesn't have to be a single leaf - an auto-expanded
        vector can be split out as one whole unit alongside other pieces."""
        POS = Variable("pos", 3)
        F = Variable("f", 1)
        data = torch.rand(5, 4)
        config = DataConfiguration(BatchAxes(5), FeatureAxes(POS * F))
        source = _source(data, config, name="Source")
        split = SplitVariables([POS, F], backend=TorchBackend, name="Split")

        graph = Graph()
        graph.add_node(source)
        graph.add_node(split)
        graph.connect(source.output_ports[0], split.input_ports[0])
        graph.setup()
        graph.run()

        assert torch.allclose(split.output_ports[0].value, data[:, :3])
        assert torch.allclose(split.output_ports[1].value, data[:, 3:])


class TestConcatVariables:
    def test_output_variable_dim_is_the_sum_not_none(self):
        """Regression: folding from an empty Variable() placeholder left
        .dim as None on the composed result, breaking get_slice()."""
        X, Y = Variable("x", 1), Variable("y", 1)
        join = ConcatVariables([X, Y], backend=TorchBackend)
        out_var = join.output_ports[0].data_configuration.variables
        assert out_var.dim == 2
        assert out_var.get_slice(X) == slice(0, 1)
        assert out_var.get_slice(Y) == slice(1, 2)

    def test_check_unique_var_keys_raises_on_duplicate_names(self):
        X, X2 = Variable("x", 1), Variable("x", 1)
        with pytest.raises(ValueError, match="not unique"):
            ConcatVariables([X, X2], backend=TorchBackend)

    def test_concatenates_real_tensors(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        join = ConcatVariables([X, Y], backend=TorchBackend)
        x_data, y_data = torch.rand(5, 1), torch.rand(5, 1)

        x_source = _source(
            x_data, DataConfiguration(BatchAxes(5), FeatureAxes(X)), name="XSource"
        )
        y_source = _source(
            y_data, DataConfiguration(BatchAxes(5), FeatureAxes(Y)), name="YSource"
        )
        graph = Graph()
        graph.add_node(x_source)
        graph.add_node(y_source)
        graph.add_node(join)
        graph.connect(x_source.output_ports[0], join.get_input_port("x"))
        graph.connect(y_source.output_ports[0], join.get_input_port("y"))
        graph.setup()
        graph.run()

        assert torch.allclose(join.output_ports[0].value, torch.cat([x_data, y_data], dim=-1))
