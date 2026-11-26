from typing import Annotated

import pytest
import torch

from qewton.algorithms.building_blocks.array_operations import (
    AsVariable,
    ConcatVariables,
    SplitVariables,
)
from qewton.backends import TensorType, TorchBackend
from qewton.config.axes import FeatureAxes, BatchAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.errors import DataConfigMismatchError
from qewton.config.variables import Variable
from qewton.constraints import MSEConstraint
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

        assert torch.allclose(graph.value_of(split.output_ports[0]), data[:, :1])
        assert torch.allclose(graph.value_of(split.output_ports[1]), data[:, 1:])

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

        assert torch.allclose(graph.value_of(split.output_ports[0]), data[:, :3])
        assert torch.allclose(graph.value_of(split.output_ports[1]), data[:, 3:])


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

        assert torch.allclose(graph.value_of(join.output_ports[0]), torch.cat([x_data, y_data], dim=-1))


class TestAsVariable:
    def test_passes_the_data_on_under_the_new_variable(self):
        T, U = Variable("t", 1), Variable("u", 1)
        data = torch.rand(5, 1)
        source = _source(
            data, DataConfiguration(BatchAxes(5), FeatureAxes(T)), name="Source"
        )
        as_u = AsVariable(U, backend=TorchBackend)

        graph = Graph()
        graph.add_node(source)
        graph.add_node(as_u)
        graph.connect(source.output_ports[0], as_u.input_ports[0])
        graph.setup()
        graph.run()

        assert as_u.output_ports[0].name == "u"
        assert graph.config_of(as_u.output_ports[0]).feature_axes.variables is U
        assert torch.equal(graph.value_of(as_u.output_ports[0]), data)

    def test_rejects_features_of_another_size(self):
        X, U = Variable("x", 2), Variable("u", 1)
        source = _source(
            torch.rand(5, 2),
            DataConfiguration(BatchAxes(5), FeatureAxes(X)),
            name="Source",
        )
        as_u = AsVariable(U, backend=TorchBackend)

        graph = Graph()
        graph.add_node(source)
        graph.add_node(as_u)
        with pytest.raises(DataConfigMismatchError):
            graph.connect(source.output_ports[0], as_u.input_ports[0])
            graph.setup()

    def _sources(self, T, U):
        u_data, t_data = torch.rand(5, 1), torch.rand(5, 1)
        u_source = _source(
            u_data, DataConfiguration(BatchAxes(5), FeatureAxes(U)), name="USource"
        )
        t_source = _source(
            t_data, DataConfiguration(BatchAxes(5), FeatureAxes(T)), name="TSource"
        )
        return u_data, t_data, u_source, t_source

    def test_a_metric_compares_data_of_the_same_variable_only(self):
        T, U = Variable("t", 1), Variable("u", 1)
        _, _, u_source, t_source = self._sources(T, U)
        mse = MSEConstraint()

        graph = Graph()
        with graph.tracker(0):
            mse(u_source(), t_source() * 2.0)
        with pytest.raises(DataConfigMismatchError):
            graph.setup()

    def test_tracked_as_variable_lets_a_metric_compare_with_a_computed_reference(self):
        """E.g. the output of a model of u against an exact solution
        computed from t."""
        T, U = Variable("t", 1), Variable("u", 1)
        u_data, t_data, u_source, t_source = self._sources(T, U)
        mse = MSEConstraint()

        graph = Graph()
        with graph.tracker(0):
            mse(u_source(), (t_source() * 2.0).as_variable(U))
        graph.setup()
        graph.run()

        expected = ((u_data - 2.0 * t_data) ** 2).sum(-1).mean()
        assert torch.allclose(graph.value_of(mse.output_ports[0]), expected)


class TestSetItem:
    def _assign(self, value):
        """Track ``n = zeros_like(x); n[:, :1] = value`` and run it."""
        from qewton.algorithms.building_blocks import ZerosLike

        X = Variable("x", 2)
        data = torch.rand(5, 2)
        source = _source(
            data, DataConfiguration(BatchAxes(5), FeatureAxes(X)), name="Source"
        )
        graph = Graph()
        with graph.tracker(0):
            normals = ZerosLike(backend=TorchBackend)(source())
            normals[:, :1] = value
        graph.setup()
        graph.run()
        return graph.value_of(normals.last_output_port)

    @pytest.mark.parametrize("value", [1.0, 1, -1.5])
    def test_assigns_a_python_number(self, value):
        result = self._assign(value)
        assert torch.all(result[:, :1] == value)
        assert torch.all(result[:, 1:] == 0.0)

    def test_assigns_a_tensor(self):
        result = self._assign(torch.full((5, 1), 2.0))
        assert torch.all(result[:, :1] == 2.0)
        assert torch.all(result[:, 1:] == 0.0)
