"""Nodes record the device of `to()` and build on it in `setup()`, so
`to()` is called once, before `setup()` (see DECISIONS.md, Devices)."""

import os

import pytest
import torch

from qewton.algorithms.building_blocks.parameters import ParameterNode
from qewton.algorithms.dl_models.fcn import FCN
from qewton.backends import TorchBackend
from qewton.config.axes import AxesDim, BatchAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.devices import cpu
from qewton.config.saving.callables import load, save
from qewton.config.variables import Variable
from qewton.constraints import MSEConstraint
from qewton.data.data_processing.normalization import StdNormalizationNode
from qewton.data.dataloaders.base import DataLoader
from qewton.data.datasets.array_data.base import ArrayLikeDataSet
from qewton.graphs.graphs import Graph
from qewton.optim import Adam, GraphBasedTrainer, OptimizationPhase

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
GPU = "cuda:0"

X, U = Variable("x", 2), Variable("u", 1)


def _model():
    return FCN(in_neurons=X, out_neurons=U, hidden_neurons=4, n_hidden_layers=1)


def _parameter_devices(node):
    return {str(p.parameters.device) for p in node.trainable_parameters}


def _loader(n=20, path=None):
    config_x = DataConfiguration(BatchAxes(AxesDim(n)), FeatureAxes(X))
    config_u = DataConfiguration(BatchAxes(AxesDim(n)), FeatureAxes(U))
    if path is None:
        data = [torch.rand(n, 2) * 5.0 + 3.0, torch.rand(n, 1)]
        data_set = ArrayLikeDataSet(data, [config_x, config_u])
    else:
        torch.save(torch.rand(n, 2), os.path.join(path, "x.pt"))
        torch.save(torch.rand(n, 1), os.path.join(path, "u.pt"))
        data_set = ArrayLikeDataSet.from_file(
            [os.path.join(path, "x.pt"), os.path.join(path, "u.pt")],
            [config_x, config_u],
            TorchBackend,
            load_on_setup=True,
        )
    return DataLoader(data_set, batch_size=10, shuffle_data=False)


class TestRecording:
    def test_node_without_data_follows_its_inputs(self):
        assert ParameterNode(shape=(2,)).device is None

    def test_data_node_defaults_to_cpu(self):
        assert _loader().device == cpu

    def test_to_records_the_device(self):
        node = ParameterNode(shape=(2,))
        node.to("cpu")
        assert node.device == "cpu"

    def test_setup_moves_to_the_device_after_building(self):
        moved = []

        class SpyParameterNode(ParameterNode):
            def to(self, device):
                moved.append((device, self._trainable_parameter is not None))
                super().to(device)

        node = SpyParameterNode(shape=(2,))
        node.to("cpu")
        node.setup()

        # first only recorded, then applied to the built parameter
        assert moved == [("cpu", False), ("cpu", True)]

    def test_setup_without_device_does_not_move(self):
        moved = []

        class SpyParameterNode(ParameterNode):
            def to(self, device):
                moved.append(device)
                super().to(device)

        SpyParameterNode(shape=(2,)).setup()
        assert not moved

    def test_graph_node_passes_its_device_to_inner_nodes(self):
        model = _model()
        model.to("cpu")
        model.setup()
        assert model._graph.nodes
        assert all(node.device == "cpu" for node in model._graph.nodes)

    def test_device_is_not_saved(self, tmp_path):
        model = _model()
        model.to("cpu")
        model.setup()
        save(model, tmp_path / "model")

        loaded = load(tmp_path / "model")
        assert loaded.device is None
        assert all(node.device is None for node in loaded._graph.nodes)

    def test_data_node_device_is_not_saved(self):
        # A whole DataLoader can not be saved yet (its random generator is
        # not serializable), so check what it would save.
        loader = _loader()
        loader.to("cpu")
        keys, _ = loader._collect_serializable_attributes()
        assert "_device" not in keys


@needs_gpu
class TestOnGPU:
    def test_model_is_built_on_the_device(self):
        model = _model()
        model.to(GPU)
        model.setup()
        assert _parameter_devices(model) == {GPU}

    def test_rebuild_stays_on_the_device(self):
        model = _model()
        model.to(GPU)
        model.setup()
        model.reset()
        model.setup()
        assert _parameter_devices(model) == {GPU}

    def test_data_processing_node_fits_on_the_device(self):
        loader = _loader()
        norm = StdNormalizationNode(data_source_node=loader)
        graph = Graph()
        graph.connect(loader.get_output_port(X), norm)
        for node in graph.nodes:
            node.to(GPU)

        graph.setup()

        data = loader.data_set.load_complete_data(data_item=0)
        assert data.is_cuda
        assert norm.mean.is_cuda
        assert torch.allclose(norm.mean, data.mean(dim=0, keepdim=True))

    def test_trainer_builds_everything_on_the_device(self, tmp_path):
        loader = _loader(path=str(tmp_path))
        model, constraint = _model(), MSEConstraint()
        graph = Graph()
        graph.connect(loader.get_output_port(X), model)
        graph.connect(model, constraint.input_1)
        graph.connect(loader.get_output_port(U), constraint.input_2)
        trainer = GraphBasedTrainer(
            [OptimizationPhase(Adam(), lr=0.01, max_iterations=2)],
            [graph],
            [constraint],
            device=GPU,
            save_path=str(tmp_path / "run"),
        )

        trainer.run(show_progress=False)

        assert all(t.is_cuda for t in loader.data_set.load_complete_data())
        assert _parameter_devices(model) == {GPU}
