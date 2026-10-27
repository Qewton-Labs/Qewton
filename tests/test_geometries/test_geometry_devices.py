"""Geometries have an explicit device that only `to()` changes.

Methods that get points compute on the points' device, methods that create
points create them on the requested device, and neither moves the geometry.
Discrete geometries warn when their data has to be copied for a call.
"""

import warnings

import numpy as np
import pytest
import torch

from qewton.backends.torch.base import get_torch_device
from qewton.config.devices import cpu, cuda
from qewton.config.variables import Variable
from qewton.geometries.base import DeviceTransferWarning, Geometry
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.circle import Circle
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.continuous.domains_2d.triangle import Triangle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.continuous.domains_3d.cylinder import Cylinder
from qewton.geometries.continuous.domains_3d.sphere import Sphere
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.geometries.discrete.index_grid_geometry import IndexGridGeometry
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.geometries.discrete.point_cloud import PointCloud

DEVICES = [cpu, cuda(0)] if torch.cuda.is_available() else [cpu]
PAIRS = [
    pytest.param(g, p, id=f"geometry_{g}-points_{p}") for g in DEVICES for p in DEVICES
]

T, X, Y = Variable("t", 1), Variable("x", 2), Variable("y", 3)


def _square():
    return Rectangle(X, [0.0, 0.0], 1.0, 1.0)


def _mesh():
    return _square().create_mesh(0.25)


CONTINUOUS = {
    "interval": lambda: Interval(T, 0.0, 2.0),
    "interval_boundary": lambda: Interval(T, 0.0, 2.0).boundary,
    "interval_left": lambda: Interval(T, 0.0, 2.0).boundary_left,
    "rectangle": _square,
    "rectangle_boundary": lambda: _square().boundary,
    "triangle": lambda: Triangle(X, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]),
    "triangle_boundary": lambda: Triangle(X, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]).boundary,
    "circle": lambda: Circle(X, [0.5, 0.5], 0.5),
    "circle_boundary": lambda: Circle(X, [0.5, 0.5], 0.5).boundary,
    "box": lambda: Box(Y, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0),
    "box_boundary": lambda: Box(Y, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0).boundary,
    "cylinder": lambda: Cylinder(Y, [0.0, 0.0, 0.0], 1.0, 1.0),
    "cylinder_boundary": lambda: Cylinder(Y, [0.0, 0.0, 0.0], 1.0, 1.0).boundary,
    "sphere": lambda: Sphere(Y, [0.0, 0.0, 0.0], 1.0),
    "sphere_boundary": lambda: Sphere(Y, [0.0, 0.0, 0.0], 1.0).boundary,
    "union": lambda: _square() + Circle(X, [1.0, 0.5], 0.5),
    "cut": lambda: _square() - Circle(X, [0.5, 0.5], 0.25),
    "intersection": lambda: _square() & Circle(X, [0.5, 0.5], 0.6),
    "product": lambda: Interval(T, 0.0, 2.0) * _square(),
    "product_boundary": lambda: Interval(T, 0.0, 2.0) * _square().boundary,
}

DISCRETE = {
    "mesh": _mesh,
    "mesh_boundary": lambda: _mesh().boundary,
    "grid": lambda: GridGeometry.build_unit_grid(X, 5),
    "point_cloud": lambda: PointCloud(X, np.random.rand(20, 2).astype(np.float32)),
    "index_grid": lambda: IndexGridGeometry(X, (4, 5)),
}

GEOMETRIES = {**CONTINUOUS, **DISCRETE}

#: References that are not part of a geometry's own data.
_NOT_OWNED = {"discretization_of", "source_geometry"}


def _owned_tensors(geometry):
    """Every tensor reachable from `geometry` through the objects it owns
    (sub-geometries, meshes, lists of those)."""
    found, seen = [], set()

    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            found.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                visit(item)
        elif isinstance(obj, (Geometry, Mesh)):
            for key, value in vars(obj).items():
                if key not in _NOT_OWNED:
                    visit(value)

    visit(geometry)
    return found


def _on(tensor, device) -> bool:
    return tensor.device == get_torch_device(device)


def _assert_all_on(geometry, device):
    off = [t.device for t in _owned_tensors(geometry) if not _on(t, device)]
    assert not off, f"tensors left on {set(off)} instead of {device}"


def _as_points(points):
    return points[0] if isinstance(points, tuple) else points


def _call_everything(geometry, device):
    """Call every method that creates or gets points; return the results of
    those the geometry implements."""
    calls = {
        "sample_random_uniform": lambda: _as_points(
            geometry.sample_random_uniform(20, device=device)
        ),
        "sample_grid": lambda: _as_points(geometry.sample_grid(20, device=device)),
    }
    results = {}
    for method, call in calls.items():
        try:
            results[method] = call()
        except NotImplementedError:
            pass
    points = results["sample_random_uniform"]
    results["contains"] = geometry.contains(points)
    if _is_boundary(geometry):
        results["normal"] = geometry.normal(points, device=device)
    return results


def _is_boundary(geometry):
    from qewton.geometries.base import BoundaryGeometry

    return isinstance(geometry, BoundaryGeometry)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_to_moves_all_data_and_records_the_device(name, device):
    geometry = GEOMETRIES[name]()
    assert geometry.to(device) is geometry
    assert geometry.device == device
    _assert_all_on(geometry, device)


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_a_new_geometry_is_on_the_cpu(name):
    geometry = GEOMETRIES[name]()
    assert geometry.device == cpu
    _assert_all_on(geometry, cpu)


@pytest.mark.parametrize("geometry_device, points_device", PAIRS)
@pytest.mark.parametrize("name", list(CONTINUOUS))
def test_continuous_methods_follow_the_points_and_keep_the_geometry(
    name, geometry_device, points_device
):
    geometry = CONTINUOUS[name]().to(geometry_device)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeviceTransferWarning)
        results = _call_everything(geometry, points_device)
    for method, result in results.items():
        assert _on(result, points_device), method
    assert geometry.device == geometry_device
    _assert_all_on(geometry, geometry_device)


@pytest.mark.parametrize("geometry_device, points_device", PAIRS)
@pytest.mark.parametrize("name", list(DISCRETE))
def test_discrete_methods_follow_the_points_and_keep_the_geometry(
    name, geometry_device, points_device
):
    geometry = DISCRETE[name]().to(geometry_device)
    if geometry_device == points_device:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeviceTransferWarning)
            results = _call_everything(geometry, points_device)
    else:
        with pytest.warns(DeviceTransferWarning):
            results = _call_everything(geometry, points_device)
    for method, result in results.items():
        assert _on(result, points_device), method
    assert geometry.device == geometry_device
    _assert_all_on(geometry, geometry_device)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_bounding_box_is_on_the_geometrys_device(name, device):
    geometry = GEOMETRIES[name]().to(device)
    try:
        box = geometry.bounding_box()
    except NotImplementedError:
        pytest.skip("no bounding box")
    assert _on(box, device)


@pytest.mark.parametrize("geometry_device, points_device", PAIRS)
@pytest.mark.parametrize("name", list(CONTINUOUS))
def test_create_mesh_keeps_the_geometry(name, geometry_device, points_device):
    geometry = CONTINUOUS[name]().to(geometry_device)
    try:
        geometry.create_mesh(0.5, device=points_device)
    except NotImplementedError:
        pytest.skip("no mesh")
    assert geometry.device == geometry_device
    _assert_all_on(geometry, geometry_device)


def test_moving_a_domain_moves_its_boundary_and_vice_versa():
    square = _square()
    boundary = square.boundary
    device = DEVICES[-1]
    boundary.to(device)
    assert square.device == device
    square.to(cpu)
    assert boundary.device == cpu
