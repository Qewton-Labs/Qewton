import pytest
from qewton.config.axes import (
    Axes,
    AxesDim,
    EllipsisDim,
    AddedDim,
    MinimumDim,
    BatchAxes,
    GeometryAxes,
    FeatureAxes,
    EllipsisAxes,
)
from qewton.config.variables import Variable
from qewton.config.errors import DataConfigMismatchError
from qewton.geometries.base import Geometry

# --- AxesDim Tests ---


def test_axes_dim_init():
    dim = AxesDim(size=10, broadcastable=True)
    assert dim.size == 10
    assert dim.broadcastable is True
    assert str(dim) == "10"


def test_axes_dim_add():
    d1 = AxesDim(10)
    d2 = AxesDim(5)
    added = d1 + d2
    assert isinstance(added, AddedDim)
    assert added.size == 15


def test_minimum_dim():
    d1 = AxesDim(10)
    d2 = AxesDim(5)
    min_dim = MinimumDim(d1, d2)
    assert min_dim.size == 5


# --- Axes Core Tests ---


def test_axes_init_basic():
    axes = Axes(10, 20, ...)
    assert len(axes.shape) == 3
    assert isinstance(axes.shape[0], AxesDim)
    assert isinstance(axes.shape[2], EllipsisDim)


def test_axes_matches():
    a1 = Axes(10, 20)
    a2 = Axes(10, 20)
    a3 = Axes(10, 30)
    assert a1.matches(a2)
    assert not a1.matches(a3)


def test_axes_is_empty():
    assert Axes().is_empty
    assert not Axes(1).is_empty


# --- Unification & Ellipsis Tests ---


# --- Specialized Axes Tests ---


def test_batch_axes():
    b = BatchAxes(32)
    assert b.shape[0].size == 32


def test_geometry_axes_init():
    geo = Geometry(shape=(128, 128))
    ga = GeometryAxes(geometry=geo)
    assert len(ga.shape) == 2
    assert ga.geometry == geo


def test_geometry_axes_from_shape():
    ga = GeometryAxes(shape=(64, 64))
    assert ga.shape[0].size == 64


def test_feature_axes_with_variable():
    v = Variable("input", 100)
    fa = FeatureAxes(variable=v)
    assert fa.shape[0].size == 100
    assert fa.variables == v


def test_feature_axes_get_slice():
    v = Variable.from_dict({"a": 5, "b": 5})
    fa = FeatureAxes(variable=v)
    slc = fa.get_variable_slice(Variable.from_dict({"a": 5}))
    assert slc == slice(0, 5, None)


def test_ellipsis_axes():
    ea = EllipsisAxes()
    assert isinstance(ea.shape[0], EllipsisDim)
    assert str(ea) == "..."


# --- Update Axes Logic ---


# --- Geometry Logic ---


# --- Edge Cases ---

