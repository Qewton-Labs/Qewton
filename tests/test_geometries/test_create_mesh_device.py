"""`create_mesh(device=...)` creates the mesh data on that device, for every
geometry that creates new mesh data. Already discrete geometries return
themselves unchanged."""

import pytest

import torch

from qewton.config.devices import cpu, cuda
from qewton.config.variables import Variable
from qewton.backends.torch.base import get_torch_device
from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.circle import Circle
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.continuous.domains_2d.triangle import Triangle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.continuous.domains_3d.cylinder import Cylinder
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.mesh_geometry import MeshGeometry

devices = [cpu, cuda(0)] if torch.cuda.is_available() else [cpu]

T = Variable("t", 1)
X = Variable("x", 2)
Y = Variable("y", 3)


def _geometries():
    interval = Interval(T, 0.0, 2.0)
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    return {
        "interval": interval,
        "interval_boundary": interval.boundary,
        "interval_left": interval.boundary_left,
        "rectangle": square,
        "rectangle_boundary": square.boundary,
        "triangle": Triangle(X, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]),
        "circle": Circle(X, [0.0, 0.0], 1.0),
        "box": Box(Y, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0),
        "box_boundary": Box(Y, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0).boundary,
        "cylinder": Cylinder(Y, [0.0, 0.0, 0.0], 1.0, 1.0),
        "space_time": interval * square,
        "time_boundary": interval * square.boundary,
    }


def _on(tensor, device) -> bool:
    return tensor.device == get_torch_device(device)


@pytest.mark.parametrize("device", devices, ids=str)
@pytest.mark.parametrize("name", list(_geometries()))
def test_mesh_data_is_created_on_the_requested_device(name, device):
    mesh_geometry = _geometries()[name].create_mesh(0.5, device=device)
    points = getattr(mesh_geometry, "discretization_points", None)
    vertices = points if points is not None else mesh_geometry.mesh.vertices
    assert _on(vertices, device)


@pytest.mark.parametrize("device", devices, ids=str)
def test_a_mesh_geometry_returns_itself(device):
    mesh_geometry = MeshGeometry(
        T, Mesh(vertices=[[0.0], [1.0], [2.0]], cells=[[0, 1], [1, 2]])
    )
    assert mesh_geometry.create_mesh(0.5, device=device) is mesh_geometry


@pytest.mark.parametrize("device", devices, ids=str)
@pytest.mark.parametrize(
    "geometry",
    [
        Interval(T, 0.0, 2.0),
        MeshGeometry(T, Mesh(vertices=[[0.0], [1.0], [2.0]], cells=[[0, 1], [1, 2]])),
    ],
    ids=["interval", "mesh_geometry"],
)
def test_mesh_mode_points_are_on_the_samplers_device(geometry, device):
    sampler = RandomUniformSampler(geometry, n_points=10)
    sampler.to(device)
    sampler.set_mesh_mode(0.5)
    try:
        assert _on(sampler.forward(), device)
    finally:
        sampler.unset_mesh_mode()
