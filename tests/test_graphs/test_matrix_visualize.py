"""Test matrix of Graph.visualize: domain × reference × error.

Every domain builds a graph with a model and a second model of the same
output Variable (the port reference), and gives reference data on its own
points. Every combination checks the same: the panels and their names,
the shared color scale and the error panel's own one, the error computed
from prediction and reference, the shared controls, and drawing with
Plotly. Combinations that are not built (EXCLUSIONS) or raise (RAISES)
are listed with their reason.
"""

import numpy as np
import pytest

from qewton.algorithms.dl_models.fcn import FCN
from qewton.config.axes import FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.geometries.discrete.point_cloud import PointCloud
from qewton.graphs.graphs import Graph
from qewton.visualization import Figure, Overlay, TimeAxis
from qewton.visualization.renderers import PlotlyRenderer

T, X, U = Variable("t", 1), Variable("x", 2), Variable("u", 1)


def _square():
    return Rectangle(X, [0.0, 0.0], 1.0, 1.0)


def _graph(sampler, in_variable):
    model, reference_model = (
        FCN(in_neurons=in_variable, hidden_neurons=4, out_neurons=U, n_hidden_layers=1)
        for _ in range(2)
    )
    graph = Graph()
    graph.connect(sampler, model)
    graph.connect(sampler, reference_model)
    graph.setup()
    return graph, model, reference_model


class Domain:
    """A graph on one kind of domain, with what visualizing it needs."""

    controls: list = []
    max_vertex_distance = 0.5

    def graph(self):
        raise NotImplementedError

    def reference_data(self):
        """Reference values and their configuration on own points."""
        raise NotImplementedError


class Square(Domain):
    def graph(self):
        return _graph(GridSampler(_square(), 20), X)

    def reference_data(self):
        points = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]], dtype=np.float32)
        config = DataConfiguration(GeometryAxes(PointCloud(X, points)), FeatureAxes(U))
        return np.array([[1.0], [2.0], [3.0]], dtype=np.float32), config


class SpaceTime(Domain):
    """A sampler on the product of time and space, drawn over space."""

    def __init__(self):
        self.controls = [TimeAxis(T)]

    def graph(self):
        return _graph(RandomUniformSampler(Interval(T, 0.0, 2.0) * _square(), 10), T * X)

    def reference_data(self):
        times = MeshGeometry(
            T, Mesh(vertices=np.array([[0.0], [0.5], [2.0]]), cells=[[0, 1], [1, 2]])
        )
        geometry = (times * _square().create_mesh(0.5)).create_mesh()
        points = np.asarray(geometry.discretization_points)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(U))
        return (points[:, :1] + points[:, 1:2]).astype(np.float32), config


class ProductSampler(SpaceTime):
    """A product of a time and a space sampler: one axes per factor."""

    max_vertex_distance = {T: 0.5, X: 0.25}

    def graph(self):
        sampler = RandomUniformSampler(Interval(T, 0.0, 2.0), 5) * RandomUniformSampler(
            _square(), 7
        )
        return _graph(sampler, T * X)


DOMAINS = {"square": Square, "space_time": SpaceTime, "product_sampler": ProductSampler}
REFERENCES = ["none", "data", "port", "callable"]
ERRORS = ["signed", "absolute", "relative", None]

#: (domain, reference) -> why it is not built.
EXCLUSIONS = {
    ("product_sampler", "data"): "reference data on a product sampler is given per "
    "factor (one GeometryAxes each), tested in test_visualize.py",
}

#: (domain, reference) -> why it raises a ValueError.
RAISES = {
    ("space_time", "port"): "a port reference runs without mesh mode: random "
    "space-time points have no time factor to step through",
}

#: The error each mode computes from prediction and reference.
ERROR_OF = {
    "signed": lambda pred, ref: pred - ref,
    "absolute": lambda pred, ref: np.abs(pred - ref),
}


def _reference(domain, kind, reference_model):
    if kind == "data":
        values, config = domain.reference_data()
        return dict(reference=values, reference_config=config)
    if kind == "port":
        return dict(reference=reference_model.output_ports[0])
    return dict(reference=lambda points: points[..., :1] * 0.5 + 0.1)


@pytest.mark.parametrize("error", ERRORS)
@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("domain_name", list(DOMAINS))
def test_visualize(domain_name, reference, error):
    if (domain_name, reference) in EXCLUSIONS:
        pytest.skip(EXCLUSIONS[(domain_name, reference)])
    if reference == "none" and error is not None:
        pytest.skip("the error needs a reference")
    domain = DOMAINS[domain_name]()
    graph, model, reference_model = domain.graph()
    options = dict(controls=domain.controls, max_vertex_distance=domain.max_vertex_distance)
    if reference != "none":
        options.update(_reference(domain, reference, reference_model), error=error)

    if (domain_name, reference) in RAISES:
        with pytest.raises(ValueError, match="not a factor"):
            graph.visualize(model.output_ports[0], **options)
        return
    layout = graph.visualize(model.output_ports[0], **options)

    plots = layout.plots
    if reference == "none":
        assert len(plots) == 1
    else:
        _check_panels(layout, error)
    for plot in plots:
        assert plot.controls[: len(domain.controls)] == domain.controls
    Figure(layout, renderer=PlotlyRenderer()).draw()


def _check_panels(layout, error):
    plots = layout.plots
    names = ["Reference", "Prediction"] + (["Error"] if error is not None else [])
    shown = [p.label for p in plots] if isinstance(layout, Overlay) else [p.title for p in plots]
    assert shown == names
    reference, prediction = plots[:2]
    assert reference.color_scale is prediction.color_scale
    assert reference.controls == prediction.controls
    if error is None:
        return
    error_plot = plots[2]
    assert error_plot.controls == reference.controls
    assert error_plot.color_scale is not reference.color_scale
    if error == "absolute":
        assert error_plot.color_scale.vmin == 0.0
    else:
        assert error_plot.color_scale.symmetric
    ref, pred, err = (np.asarray(p.data, dtype=float) for p in plots)
    if error in ERROR_OF:
        np.testing.assert_allclose(err, ERROR_OF[error](pred, ref), atol=1e-5)
    else:
        near_zero = np.abs(ref) < 1e-8 * np.abs(ref).max()
        np.testing.assert_allclose(
            err[~near_zero], ((pred - ref) / ref)[~near_zero], rtol=1e-4, atol=1e-5
        )
