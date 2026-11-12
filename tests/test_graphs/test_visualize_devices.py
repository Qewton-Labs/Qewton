"""Graph.visualize() works for every kind of plot, whether the model lives on
the CPU or a GPU, with and without an explicit `device`."""

import numpy as np
import pytest

from qewton.algorithms.building_blocks.activation_functions import Tanh
from qewton.algorithms.dl_models.fcn import FCN
from qewton.config.axes import FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
import torch

from qewton.config.devices import cpu, cuda
from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.graphs.graphs import Graph
from qewton.optim.base import EvaluationPhase
from qewton.visualization.figure import Figure
from qewton.visualization import SliderAxis, TimeAxis

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

T, X, Y, U = Variable("t", 1), Variable("x", 2), Variable("y", 3), Variable("u", 1)


def _graph(sampler, in_variable):
    model = FCN(
        in_neurons=in_variable, out_neurons=U, hidden_neurons=4,
        n_hidden_layers=1, activation=Tanh,
    )
    graph = Graph()
    graph.connect(sampler, model)
    graph.setup()
    return graph, model


def _integrator():
    graph, model = _graph(GridSampler(Interval(T, 0.0, 2.0), 50), T)
    return graph, model, dict(reference=lambda t: t**2)


def _field_2d():
    graph, model = _graph(RandomUniformSampler(Rectangle(X, [0.0, 0.0], 1.0, 1.0), 10), X)
    return graph, model, {}


def _boundary():
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    graph, model = _graph(RandomUniformSampler(square.boundary, 10), X)
    return graph, model, {}


def _box():
    graph, model = _graph(RandomUniformSampler(Box(Y, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0), 10), Y)
    return graph, model, dict(max_vertex_distance=0.5)


def _space_time():
    domain = Interval(T, 0.0, 2.0) * Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    graph, model = _graph(RandomUniformSampler(domain, 10), T * X)
    return graph, model, dict(controls=[TimeAxis(T)], max_vertex_distance=0.5)


def _product_sampler():
    sampler = RandomUniformSampler(Interval(T, 0.0, 2.0), 5) * RandomUniformSampler(
        Rectangle(X, [0.0, 0.0], 1.0, 1.0), 7
    )
    graph, model = _graph(sampler, T * X)
    return graph, model, dict(controls=[TimeAxis(T)], max_vertex_distance=0.5)


def _data_reference():
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    graph, model = _graph(RandomUniformSampler(Interval(T, 0.0, 2.0) * square, 10), T * X)
    times = np.array([0.0, 0.5, 2.0])
    time_mesh = MeshGeometry(T, Mesh(vertices=times[:, None], cells=[[0, 1], [1, 2]]))
    reference_geometry = (time_mesh * square.create_mesh(0.5)).create_mesh()
    points = np.asarray(reference_geometry.discretization_points)
    config = DataConfiguration(GeometryAxes(reference_geometry), FeatureAxes(U))
    return graph, model, dict(
        reference=points[:, :1], reference_config=config, controls=[TimeAxis(T)]
    )


CASES = {
    "integrator": _integrator,
    "field_2d": _field_2d,
    "boundary": _boundary,
    "box_3d": _box,
    "space_time": _space_time,
    "product_sampler": _product_sampler,
    "data_reference": _data_reference,
}

MODEL_DEVICES = [pytest.param(cpu, id="model_cpu"), pytest.param(cuda(0), id="model_cuda", marks=needs_gpu)]
VISUALIZE_DEVICES = [
    pytest.param(None, id="no_device"),
    pytest.param(cpu, id="device_cpu"),
    pytest.param(cuda(0), id="device_cuda", marks=needs_gpu),
]


@pytest.mark.parametrize("visualize_device", VISUALIZE_DEVICES)
@pytest.mark.parametrize("model_device", MODEL_DEVICES)
@pytest.mark.parametrize("case", list(CASES))
def test_every_plot_works_on_every_device(case, model_device, visualize_device):
    graph, model, kwargs = CASES[case]()
    for node in graph.nodes:  # like a trainer: the whole graph on one device
        node.to(model_device)
    graph.run(EvaluationPhase.TRAIN)  # one step on that device, like training
    if visualize_device is not None:
        kwargs["device"] = visualize_device
    layout = graph.visualize(model.output_ports[0], **kwargs)
    for plot in layout.plots:
        assert np.all(np.isfinite(np.asarray(plot.data)))
    Figure(layout).draw()


@needs_gpu
@pytest.mark.xfail(
    reason="Without `device`, visualize() meshes on the sampler's device, which "
    "differs from the model's here. Needs a device property on nodes.",
    strict=True,
)
def test_model_on_gpu_and_sampler_on_cpu_without_device():
    graph, model, kwargs = _field_2d()
    model.to(cuda(0))
    graph.visualize(model.output_ports[0], **kwargs)


@pytest.mark.parametrize("model_device", MODEL_DEVICES)
def test_data_reference_after_training_with_another_variable_order(model_device):
    """Training samples on the model's device; the reference data is on the
    CPU and lists its factors in another order (alpha, x) than the sampler
    and the model (x, alpha)."""
    import math

    from qewton.constraints.pinn_constraint import PINNConstraint  # noqa: F401
    from qewton.geometries.discrete.product_mesh_geometry import ProductDiscreteGeometry
    from qewton.graphs.pipelines.pinn_pipeline import PINNPipeline
    from qewton.optim import Adam, GraphBasedTrainer, OptimizationPhase

    A = Variable("alpha", 1)
    V = Variable("v", 1)
    omega = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    alphas = Interval(A, math.pi / 4.0, math.pi / 2.0)
    model = FCN(in_neurons=X * A, out_neurons=V, hidden_neurons=4,
                n_hidden_layers=1, activation=Tanh)

    def residual(v: V, x: X):
        return v.gradient(x)

    pipeline = PINNPipeline(RandomUniformSampler(omega * alphas, 20), [model],
                            residual=residual)
    GraphBasedTrainer(
        optimization_phases=OptimizationPhase(optimizer=Adam(), lr=1e-3, max_iterations=2),
        graphs=[pipeline], training_objectives=[pipeline.constraint], device=model_device,
    ).run(show_progress=False)

    angles = np.array([[math.pi / 4.0], [math.pi / 3.0], [math.pi / 2.0]], dtype=np.float32)
    g1, g2 = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 4), indexing="ij")
    space = np.stack([g1, g2], axis=-1).astype(np.float32)
    reference_geometry = ProductDiscreteGeometry(
        [alphas.discretize(angles), omega.discretize(space)], variable=X * A
    )
    reference = np.zeros((3, 5, 4, 1), dtype=np.float32)
    config = DataConfiguration(GeometryAxes(reference_geometry), FeatureAxes(V))
    layout = pipeline.visualize(
        model.output_ports[0], reference=reference, reference_config=config,
        controls=[SliderAxis(A)],
    )
    assert len(layout.plots) == 3
    Figure(layout).draw()



@pytest.mark.parametrize("model_device", MODEL_DEVICES)
def test_data_reference_on_a_mesh_domain_does_not_copy_the_mesh(model_device):
    """The domain check runs where the domain lives, so a reference on the
    CPU does not make a mesh on the GPU copy itself."""
    import warnings

    from qewton.geometries.base import DeviceTransferWarning

    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    graph, model = _graph(RandomUniformSampler(square.create_mesh(0.25), 10), X)
    for node in graph.nodes:
        node.to(model_device)
    reference_geometry = square.create_mesh(0.5)
    config = DataConfiguration(GeometryAxes(reference_geometry), FeatureAxes(U))
    reference = np.zeros((len(reference_geometry.mesh.vertices), 1), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeviceTransferWarning)
        layout = graph.visualize(
            model.output_ports[0], reference=reference, reference_config=config
        )
    assert len(layout.plots) == 3
