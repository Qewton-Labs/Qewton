"""Test matrix of sizes set by data sources: source × check.

A data loader and a sampler feed the FCN of `model_kinds`. Their size
HyperParameter (`batch_size`) is changed after the graph was set up; the
configurations behind them must follow without another `connect` or
`setup` (config D2). The outputs of a source share one batch axes (config D1), a
smaller last batch runs (config D3), and in mesh mode the dims bound to the
number of points have the number of mesh points, until mesh mode ends
(config S3). Combinations that do not hold yet are listed in GAPS with their
reason.
"""

import pytest
import torch
from model_kinds import DOMAIN, MODELS, U, X

from qewton.config.axes import BatchAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.data import ArrayLikeDataSet, DataLoader
from qewton.data.dataloaders.sampler.point_sampler import discretization_mode
from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler
from qewton.graphs.graphs import Graph


def _data_loader(n_samples=20, batch_size=4):
    x, u = torch.rand(n_samples, 2), torch.rand(n_samples, 1)
    configs = [
        DataConfiguration(BatchAxes(n_samples), FeatureAxes(X)),
        DataConfiguration(BatchAxes(n_samples), FeatureAxes(U)),
    ]
    data_set = ArrayLikeDataSet(data=[x, u], data_configs=configs)
    return DataLoader(data_set, batch_size=batch_size, splitting_ratio=(1.0, 0.0, 0.0))


SOURCES = {
    "data_loader": lambda: _data_loader(),
    "sampler": lambda: RandomUniformSampler(DOMAIN, n_points=4),
    "sampler_with_normals": lambda: RandomUniformSampler(
        DOMAIN.boundary, n_points=4, compute_normals=True
    ),
}


def _graph(source):
    model = MODELS["graph_node"]()
    graph = Graph()
    graph.connect(source.get_output_port(X), model)
    graph.setup()
    return graph, model


def _size_parameter(source):
    return next(h for h in source.hyperparameters if h.name == "batch_size")


def _batch_size_behind(source, graph, model):
    return model.output_ports[0].get_data_configuration(graph).shape[0]


def check_size_change_reaches_the_model(source, graph, model):
    _size_parameter(source).set_value(9)
    assert _batch_size_behind(source, graph, model) == 9


def check_outputs_share_the_batch_axes(source, graph, model):
    if len(source.output_ports) < 2:
        pytest.skip("one output only")
    first, second = (
        port.get_data_configuration(graph) or port.data_configuration
        for port in source.output_ports[:2]
    )
    assert first.axes[0] is second.axes[0]


def check_a_smaller_last_batch_runs(source, graph, model):
    if not isinstance(source, DataLoader):
        pytest.skip("only a data loader has a last batch")
    source.set_mode(source.mode)
    for _ in range(len(source) // source.batch_size + 1):
        graph.run()


def check_mesh_mode_sets_the_number_of_points(source, graph, model):
    if isinstance(source, DataLoader):
        pytest.skip("only a sampler has a mesh mode")
    if source.compute_normals:
        pytest.skip("mesh mode can not produce normals")
    with discretization_mode([source], 0.25):
        graph.run()
        n_mesh_points = len(graph.value_of(model.output_ports[0]))
        assert n_mesh_points != 4
        assert _batch_size_behind(source, graph, model) == n_mesh_points
    assert _batch_size_behind(source, graph, model) == 4


CHECKS = {
    "size_change_reaches_the_model": check_size_change_reaches_the_model,
    "outputs_share_the_batch_axes": check_outputs_share_the_batch_axes,
    "a_smaller_last_batch_runs": check_a_smaller_last_batch_runs,
    "mesh_mode_sets_the_number_of_points": check_mesh_mode_sets_the_number_of_points,
}

GAPS: dict = {}


def _cases():
    for source in SOURCES:
        for check in CHECKS:
            reason = GAPS.get((source, check)) or GAPS.get((None, check))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            yield pytest.param(source, check, marks=marks, id=f"{source}-{check}")


@pytest.mark.parametrize("source_name, check", _cases())
def test_sizes(source_name, check):
    source = SOURCES[source_name]()
    graph, model = _graph(source)
    CHECKS[check](source, graph, model)
