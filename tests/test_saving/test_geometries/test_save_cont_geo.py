import inspect
import pytest

from qewton.config.saving.callables import save, load
from qewton.backends.base import ComputingBackend
from qewton.geometries.continuous.domain_operations.cut import CutGeometry
from qewton.geometries.continuous.domain_operations.intersection import (
    IntersectionGeometry,
)
from qewton.geometries.continuous.domain_operations.union import UnionGeometry
from qewton.geometries.product import ProductGeometry
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.parallelogram import Parallelogram
from qewton.geometries.continuous.domains_2d.circle import Circle
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.continuous.domains_2d.triangle import Triangle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.continuous.domains_3d.sphere import Sphere
from qewton.geometries.continuous.domains_3d.cylinder import Cylinder
from qewton.config.variables import Variable


def all_subclasses(cls):
    result = []
    for sub_cls in cls.__subclasses__():
        if not inspect.isabstract(sub_cls) and hasattr(sub_cls, "math"):
            result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(ComputingBackend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_interval(tmp_path, backend):
    X = Variable("x", 1)
    interval = Interval(X, 0.0, 1.0, backend=backend)
    save(interval, tmp_path / "interval_save", replace=True)
    loaded_interval = load(tmp_path / "interval_save")
    assert isinstance(loaded_interval, Interval)
    assert loaded_interval.lower_bound == interval.lower_bound
    assert loaded_interval.upper_bound == interval.upper_bound


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_interval_boundary(tmp_path, backend):
    X = Variable("x", 1)
    interval = Interval(X, 0.0, 1.0, backend=backend)
    save(interval.boundary, tmp_path / "interval_boundary_save", replace=True)
    loaded_interval_bc = load(tmp_path / "interval_boundary_save")
    assert isinstance(loaded_interval_bc, type(interval.boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_circle(tmp_path, backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    save(circle, tmp_path / "circle_save", replace=True)
    loaded_circle = load(tmp_path / "circle_save")
    assert isinstance(loaded_circle, Circle)
    assert backend.math.allclose(loaded_circle.center, circle.center)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_circle_boundary(tmp_path, backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    save(circle.boundary, tmp_path / "circle_boundary_save", replace=True)
    loaded_circle_bc = load(tmp_path / "circle_boundary_save")
    assert isinstance(loaded_circle_bc, type(circle.boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_parallelogram(tmp_path, backend):
    X = Variable("x", 2)
    para = Parallelogram(X, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    save(para, tmp_path / "parallelogram_save", replace=True)
    loaded = load(tmp_path / "parallelogram_save")
    assert isinstance(loaded, Parallelogram)
    assert backend.math.allclose(loaded.origin, para.origin)
    assert backend.math.allclose(loaded.corner_1, para.corner_1)
    assert backend.math.allclose(loaded.corner_2, para.corner_2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_parallelogram_boundary(tmp_path, backend):
    X = Variable("x", 2)
    para = Parallelogram(X, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    save(para.boundary, tmp_path / "parallelogram_boundary_save", replace=True)
    loaded_bc = load(tmp_path / "parallelogram_boundary_save")
    assert isinstance(loaded_bc, type(para.boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_rectangle(tmp_path, backend):
    X = Variable("x", 2)
    rect = Rectangle(X, [0.0, 0.0], 2.0, 3.0, backend=backend)
    save(rect, tmp_path / "rectangle_save", replace=True)
    loaded = load(tmp_path / "rectangle_save")
    assert isinstance(loaded, Rectangle)
    assert backend.math.isclose(
        loaded.corner_1[0] - loaded.origin[0], backend.build_tensor([2.0])
    )
    assert backend.math.isclose(
        loaded.corner_2[1] - loaded.origin[1], backend.build_tensor([3.0])
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_rectangle_boundary(tmp_path, backend):
    X = Variable("x", 2)
    rect = Rectangle(X, [0.0, 0.0], 2.0, 3.0, backend=backend)
    save(rect.boundary, tmp_path / "rectangle_boundary_save", replace=True)
    loaded_bc = load(tmp_path / "rectangle_boundary_save")
    assert isinstance(loaded_bc, type(rect.boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_triangle(tmp_path, backend):
    X = Variable("x", 2)
    tri = Triangle(X, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    save(tri, tmp_path / "triangle_save", replace=True)
    loaded = load(tmp_path / "triangle_save")
    assert isinstance(loaded, Triangle)
    assert backend.math.allclose(loaded.origin, tri.origin)
    assert backend.math.allclose(loaded.corner_1, tri.corner_1)
    assert backend.math.allclose(loaded.corner_2, tri.corner_2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_triangle_boundary(tmp_path, backend):
    X = Variable("x", 2)
    tri = Triangle(X, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    save(tri.boundary, tmp_path / "triangle_boundary_save", replace=True)
    loaded_bc = load(tmp_path / "triangle_boundary_save")
    assert isinstance(loaded_bc, type(tri.boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_box(tmp_path, backend):
    X = Variable("x", 3)
    box = Box(X, [0.0, 0.0, 0.0], 1.0, 2.0, 3.0, backend=backend)
    save(box, tmp_path / "box_save", replace=True)
    loaded = load(tmp_path / "box_save")
    assert isinstance(loaded, Box)
    assert loaded.width == box.width
    assert loaded.height == box.height
    assert loaded.depth == box.depth


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_box_boundary(tmp_path, backend):
    X = Variable("x", 3)
    box = Box(X, [0.0, 0.0, 0.0], 1.0, 2.0, 3.0, backend=backend)
    save(box.boundary, tmp_path / "box_boundary_save", replace=True)
    loaded_bc = load(tmp_path / "box_boundary_save")
    assert isinstance(loaded_bc, type(box.boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_sphere(tmp_path, backend):
    X = Variable("x", 3)
    sphere = Sphere(X, [0.0, 0.0, 0.0], 1.0, backend=backend)
    save(sphere, tmp_path / "sphere_save", replace=True)
    loaded = load(tmp_path / "sphere_save")
    assert isinstance(loaded, Sphere)
    assert backend.math.allclose(loaded.center, sphere.center)
    assert backend.math.allclose(loaded.radius, sphere.radius)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_sphere_boundary(tmp_path, backend):
    X = Variable("x", 3)
    sphere = Sphere(X, [0.0, 0.0, 0.0], 1.0, backend=backend)
    save(sphere.boundary, tmp_path / "sphere_boundary_save", replace=True)
    loaded_bc = load(tmp_path / "sphere_boundary_save")
    assert isinstance(loaded_bc, type(sphere.boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_cylinder(tmp_path, backend):
    X = Variable("x", 3)
    cyl = Cylinder(X, [0.0, 0.0, 0.0], 1.0, 2.0, backend=backend)
    save(cyl, tmp_path / "cylinder_save", replace=True)
    loaded = load(tmp_path / "cylinder_save")
    assert isinstance(loaded, Cylinder)
    assert backend.math.allclose(loaded.center, cyl.center)
    assert backend.math.allclose(loaded.radius, cyl.radius)
    assert backend.math.allclose(loaded.height, cyl.height)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_cylinder_boundary(tmp_path, backend):
    X = Variable("x", 3)
    cyl = Cylinder(X, [0.0, 0.0, 0.0], 1.0, 2.0, backend=backend)
    save(cyl.boundary, tmp_path / "cylinder_boundary_save", replace=True)
    loaded_bc = load(tmp_path / "cylinder_boundary_save")
    assert isinstance(loaded_bc, type(cyl.boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_cut_geometry(tmp_path, backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [-0.5, -0.5], 1.0, 1.0, backend=backend)
    cut = CutGeometry(circle, rect)
    save(cut, tmp_path / "cut_geometry_save", replace=True)
    loaded = load(tmp_path / "cut_geometry_save")
    assert isinstance(loaded, CutGeometry)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_cut_geometry_boundary(tmp_path, backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [-0.5, -0.5], 1.0, 1.0, backend=backend)
    cut = CutGeometry(circle, rect)
    save(cut.boundary, tmp_path / "cut_geometry_boundary_save", replace=True)
    loaded_bc = load(tmp_path / "cut_geometry_boundary_save")
    assert isinstance(loaded_bc, type(cut.boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_intersection_geometry(tmp_path, backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [-0.5, -0.5], 1.0, 1.0, backend=backend)
    inter = IntersectionGeometry(circle, rect)
    save(inter, tmp_path / "intersection_geometry_save", replace=True)
    loaded = load(tmp_path / "intersection_geometry_save")
    assert isinstance(loaded, IntersectionGeometry)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_intersection_geometry_boundary(tmp_path, backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [-0.5, -0.5], 1.0, 1.0, backend=backend)
    inter = IntersectionGeometry(circle, rect)
    save(inter.boundary, tmp_path / "intersection_geometry_boundary_save", replace=True)
    loaded_bc = load(tmp_path / "intersection_geometry_boundary_save")
    assert isinstance(loaded_bc, type(inter.boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_union_geometry(tmp_path, backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [0.5, -0.5], 1.0, 1.0, backend=backend)
    union = UnionGeometry(circle, rect)
    save(union, tmp_path / "union_geometry_save", replace=True)
    loaded = load(tmp_path / "union_geometry_save")
    assert isinstance(loaded, UnionGeometry)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_union_geometry_boundary(tmp_path, backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [0.5, -0.5], 1.0, 1.0, backend=backend)
    union = UnionGeometry(circle, rect)
    save(union.boundary, tmp_path / "union_geometry_boundary_save", replace=True)
    loaded_bc = load(tmp_path / "union_geometry_boundary_save")
    assert isinstance(loaded_bc, type(union.boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_product_geometry(tmp_path, backend):
    X = Variable("x", 1)
    Y = Variable("y", 1)
    interval_x = Interval(X, 0.0, 1.0, backend=backend)
    interval_y = Interval(Y, 0.0, 1.0, backend=backend)
    product = ProductGeometry(interval_x, interval_y)
    save(product, tmp_path / "product_geometry_save", replace=True)
    loaded = load(tmp_path / "product_geometry_save")
    assert isinstance(loaded, ProductGeometry)
