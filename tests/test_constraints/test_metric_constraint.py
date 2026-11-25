from typing import Annotated

import pytest
import torch

from qewton.backends import TensorType, TorchBackend
from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
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


def _mse(prediction, target):
    return ((prediction - target) ** 2).sum(-1).mean()


U = Variable("u", 1)


def _config(*sizes):
    """A batch, (points,) and features of u."""
    *outer, features = sizes
    return DataConfiguration(*(BatchAxes(s) for s in outer), FeatureAxes(U))


class TestMSEConstraintTarget:
    @pytest.mark.parametrize("target", [0.0, 0, 1.5])
    def test_a_python_number_is_compared_with_every_value(self, target):
        data = torch.rand(8, 5, 1)
        source = _source(data, _config(8, 5, 1), name="Source")
        mse = MSEConstraint()

        graph = Graph()
        with graph.tracker(0):
            mse(source(), target)
        graph.setup()
        graph.run()

        assert torch.allclose(graph.value_of(mse.output_ports[0]), _mse(data, target))

    def test_a_python_number_after_slicing_the_prediction(self):
        """E.g. an initial condition ``u[:, 0]`` compared with 0.0."""
        data = torch.rand(8, 5, 1)
        config = DataConfiguration(BatchAxes(8), GeometryAxes(shape=(5,)), FeatureAxes(U))
        source = _source(data, config, name="Source")
        mse = MSEConstraint()

        graph = Graph()
        with graph.tracker(0):
            mse(source()[:, 0], 0.0)
        graph.setup()
        graph.run()

        assert torch.allclose(graph.value_of(mse.output_ports[0]), _mse(data[:, 0], 0.0))

    def test_data_of_the_same_configuration_is_compared(self):
        prediction, target = torch.rand(8, 1), torch.rand(8, 1)
        p_source = _source(prediction, _config(8, 1), name="Prediction")
        t_source = _source(target, _config(8, 1), name="Target")
        mse = MSEConstraint()

        graph = Graph()
        with graph.tracker(0):
            mse(p_source(), t_source())
        graph.setup()
        graph.run()

        assert torch.allclose(graph.value_of(mse.output_ports[0]), _mse(prediction, target))

    def test_data_of_other_dims_is_rejected_not_broadcast(self):
        """Data of shape (8, 5, 1) against (8, 1) would broadcast; the
        target must have the dims of the prediction."""
        p_source = _source(torch.rand(8, 5, 1), _config(8, 5, 1), name="Prediction")
        t_source = _source(torch.rand(8, 1), _config(8, 1), name="Target")
        mse = MSEConstraint()

        graph = Graph()
        with pytest.raises(DataConfigMismatchError):
            with graph.tracker(0):
                mse(p_source(), t_source())
            graph.setup()
