import inspect
import pytest

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
def test_save_and_load_interval(backend):
    X = Variable("x", 1)
    interval = Interval(X, 0.0, 1.0, backend=backend)
    save_config = interval.save()
    loaded_interval = Interval.load(save_config)
    assert isinstance(loaded_interval, Interval)
    assert loaded_interval.lower_bound == interval.lower_bound
    assert loaded_interval.upper_bound == interval.upper_bound


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_interval_boundary(backend):
    X = Variable("x", 1)
    interval = Interval(X, 0.0, 1.0, backend=backend)
    int_bc_type = type(interval.boundary)
    save_config = interval.boundary.save()
    loaded_interval_bc = int_bc_type.load(save_config)
    assert isinstance(loaded_interval_bc, int_bc_type)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_circle(backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    save_config = circle.save()
    loaded_circle = Circle.load(save_config)
    assert isinstance(loaded_circle, Circle)
    assert backend.math.allclose(loaded_circle.center, circle.center)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_circle_boundary(backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    circle_bc_type = type(circle.boundary)
    save_config = circle.boundary.save()
    loaded_circle_bc = circle_bc_type.load(save_config)
    assert isinstance(loaded_circle_bc, circle_bc_type)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_parallelogram(backend):
    X = Variable("x", 2)
    para = Parallelogram(X, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    save_config = para.save()
    loaded = Parallelogram.load(save_config)
    assert isinstance(loaded, Parallelogram)
    assert backend.math.allclose(loaded.origin, para.origin)
    assert backend.math.allclose(loaded.corner_1, para.corner_1)
    assert backend.math.allclose(loaded.corner_2, para.corner_2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_parallelogram_boundary(backend):
    X = Variable("x", 2)
    para = Parallelogram(X, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    bc_type = type(para.boundary)
    save_config = para.boundary.save()
    loaded_bc = bc_type.load(save_config)
    assert isinstance(loaded_bc, bc_type)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_rectangle(backend):
    X = Variable("x", 2)
    rect = Rectangle(X, [0.0, 0.0], 2.0, 3.0, backend=backend)
    save_config = rect.save()
    loaded = Rectangle.load(save_config)
    assert isinstance(loaded, Rectangle)
    assert backend.math.isclose(
        loaded.corner_1[0] - loaded.origin[0], backend.build_tensor([2.0])
    )
    assert backend.math.isclose(
        loaded.corner_2[1] - loaded.origin[1], backend.build_tensor([3.0])
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_rectangle_boundary(backend):
    X = Variable("x", 2)
    rect = Rectangle(X, [0.0, 0.0], 2.0, 3.0, backend=backend)
    bc_type = type(rect.boundary)
    save_config = rect.boundary.save()
    loaded_bc = bc_type.load(save_config)
    assert isinstance(loaded_bc, bc_type)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_triangle(backend):
    X = Variable("x", 2)
    tri = Triangle(X, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    save_config = tri.save()
    loaded = Triangle.load(save_config)
    assert isinstance(loaded, Triangle)
    assert backend.math.allclose(loaded.origin, tri.origin)
    assert backend.math.allclose(loaded.corner_1, tri.corner_1)
    assert backend.math.allclose(loaded.corner_2, tri.corner_2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_triangle_boundary(backend):
    X = Variable("x", 2)
    tri = Triangle(X, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    bc_type = type(tri.boundary)
    save_config = tri.boundary.save()
    loaded_bc = bc_type.load(save_config)
    assert isinstance(loaded_bc, bc_type)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_box(backend):
    X = Variable("x", 3)
    box = Box(X, [0.0, 0.0, 0.0], 1.0, 2.0, 3.0, backend=backend)
    save_config = box.save()
    loaded = Box.load(save_config)
    assert isinstance(loaded, Box)
    assert loaded.width == box.width
    assert loaded.height == box.height
    assert loaded.depth == box.depth


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_box_boundary(backend):
    X = Variable("x", 3)
    box = Box(X, [0.0, 0.0, 0.0], 1.0, 2.0, 3.0, backend=backend)
    bc_type = type(box.boundary)
    save_config = box.boundary.save()
    loaded_bc = bc_type.load(save_config)
    assert isinstance(loaded_bc, bc_type)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_sphere(backend):
    X = Variable("x", 3)
    sphere = Sphere(X, [0.0, 0.0, 0.0], 1.0, backend=backend)
    save_config = sphere.save()
    loaded = Sphere.load(save_config)
    assert isinstance(loaded, Sphere)
    assert backend.math.allclose(loaded.center, sphere.center)
    assert backend.math.allclose(loaded.radius, sphere.radius)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_sphere_boundary(backend):
    X = Variable("x", 3)
    sphere = Sphere(X, [0.0, 0.0, 0.0], 1.0, backend=backend)
    bc_type = type(sphere.boundary)
    save_config = sphere.boundary.save()
    loaded_bc = bc_type.load(save_config)
    assert isinstance(loaded_bc, bc_type)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_cylinder(backend):
    X = Variable("x", 3)
    cyl = Cylinder(X, [0.0, 0.0, 0.0], 1.0, 2.0, backend=backend)
    save_config = cyl.save()
    loaded = Cylinder.load(save_config)
    assert isinstance(loaded, Cylinder)
    assert backend.math.allclose(loaded.center, cyl.center)
    assert backend.math.allclose(loaded.radius, cyl.radius)
    assert backend.math.allclose(loaded.height, cyl.height)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_cylinder_boundary(backend):
    X = Variable("x", 3)
    cyl = Cylinder(X, [0.0, 0.0, 0.0], 1.0, 2.0, backend=backend)
    bc_type = type(cyl.boundary)
    save_config = cyl.boundary.save()
    loaded_bc = bc_type.load(save_config)
    assert isinstance(loaded_bc, bc_type)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_cut_geometry(backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [-0.5, -0.5], 1.0, 1.0, backend=backend)
    cut = CutGeometry(circle, rect)
    save_config = cut.save()
    loaded = CutGeometry.load(save_config)
    assert isinstance(loaded, CutGeometry)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_cut_geometry_boundary(backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [-0.5, -0.5], 1.0, 1.0, backend=backend)
    cut = CutGeometry(circle, rect)
    bc_type = type(cut.boundary)
    save_config = cut.boundary.save()
    loaded_bc = bc_type.load(save_config)
    assert isinstance(loaded_bc, bc_type)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_intersection_geometry(backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [-0.5, -0.5], 1.0, 1.0, backend=backend)
    inter = IntersectionGeometry(circle, rect)
    save_config = inter.save()
    loaded = IntersectionGeometry.load(save_config)
    assert isinstance(loaded, IntersectionGeometry)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_intersection_geometry_boundary(backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [-0.5, -0.5], 1.0, 1.0, backend=backend)
    inter = IntersectionGeometry(circle, rect)
    bc_type = type(inter.boundary)
    save_config = inter.boundary.save()
    loaded_bc = bc_type.load(save_config)
    assert isinstance(loaded_bc, bc_type)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_union_geometry(backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [0.5, -0.5], 1.0, 1.0, backend=backend)
    union = UnionGeometry(circle, rect)
    save_config = union.save()
    loaded = UnionGeometry.load(save_config)
    assert isinstance(loaded, UnionGeometry)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_union_geometry_boundary(backend):
    X = Variable("x", 2)
    circle = Circle(X, [0.0, 0.0], 1.0, backend=backend)
    rect = Rectangle(X, [0.5, -0.5], 1.0, 1.0, backend=backend)
    union = UnionGeometry(circle, rect)
    bc_type = type(union.boundary)
    save_config = union.boundary.save()
    loaded_bc = bc_type.load(save_config)
    assert isinstance(loaded_bc, bc_type)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_product_geometry(backend):
    X = Variable("x", 1)
    Y = Variable("y", 1)
    interval_x = Interval(X, 0.0, 1.0, backend=backend)
    interval_y = Interval(Y, 0.0, 1.0, backend=backend)
    product = ProductGeometry(interval_x, interval_y)
    save_config = product.save()
    loaded = ProductGeometry.load(save_config)
    assert isinstance(loaded, ProductGeometry)
