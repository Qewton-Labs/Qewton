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


def test_axes_dim_unify_identical():
    d1 = AxesDim(10)
    d2 = AxesDim(10)
    res1, res2 = d1.unify_with(d2)
    assert res1.size == 10
    assert res2.size == 10


def test_axes_dim_unify_none():
    d1 = AxesDim(None)
    d2 = AxesDim(20)
    res1, res2 = d1.unify_with(d2)
    assert res1.size == 20
    assert res2.size == 20


def test_axes_dim_unify_broadcast():
    d1 = AxesDim(1)
    d2 = AxesDim(5)
    res1, _ = d1.unify_with(d2)
    assert res1.size == 5


def test_axes_dim_unify_mismatch_raises():
    d1 = AxesDim(10)
    d2 = AxesDim(20)
    with pytest.raises(DataConfigMismatchError):
        d1.unify_with(d2)


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


def test_axes_remove_dim():
    d1 = AxesDim(10)
    d2 = AxesDim(20)
    axes = Axes(d1, d2)
    axes.remove_dim(d1)
    assert axes.shape == (d2,)


# --- Unification & Ellipsis Tests ---


def test_axes_unify_simple():
    a1 = Axes(10, 20)
    a2 = Axes(10, 20)
    map1, _ = a1.unify_with(a2)
    assert len(map1) == 2
    # Keys are the original AxesDim objects
    for dim in a1.shape:
        assert map1[dim].size == dim.size


def test_axes_unify_with_ellipsis_right():
    # (10, ...) and (10, 20, 30)
    a1 = Axes(10, ...)
    a2 = Axes(10, 20, 30)
    map1, _ = a1.unify_with(a2)

    # map1[ellipsis] should contain [20, 30]
    ell_dim = [d for d in a1.shape if isinstance(d, EllipsisDim)][0]
    assert len(map1[ell_dim]) == 2
    assert map1[ell_dim][0].size == 20
    assert map1[ell_dim][1].size == 30


def test_axes_unify_with_ellipsis_left():
    # (..., 30) and (10, 20, 30)
    a1 = Axes(..., 30)
    a2 = Axes(10, 20, 30)
    map1, _ = a1.unify_with(a2)

    ell_dim = [d for d in a1.shape if isinstance(d, EllipsisDim)][0]
    assert len(map1[ell_dim]) == 2
    assert map1[ell_dim][0].size == 10


def test_axes_unify_complex_ellipsis_overlap():
    # Smallest matching shape: (5, ...) and (..., 5) -> (5,)
    # Note: Logic in _match_remainder tries to unify matching neighbors
    d1_5 = AxesDim(5)
    d2_5 = AxesDim(5)
    a1 = Axes(d1_5, ...)
    a2 = Axes(..., d2_5)

    map1, map2 = a1.unify_with(a2)
    # Check if they unified to the same dimension
    assert map1[d1_5] == map2[d2_5]


def test_axes_unify_mismatch_raises():
    a1 = Axes(10, 20)
    a2 = Axes(10, 25)
    with pytest.raises(DataConfigMismatchError):
        a1.unify_with(a2)


def test_axes_unify_different_base_types_raises():
    a1 = BatchAxes(10)
    a2 = FeatureAxes(shape=(10,))
    with pytest.raises(DataConfigMismatchError):
        a1.unify_with(a2)


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


def test_geometry_axes_unify():
    ga1 = GeometryAxes(shape=(AxesDim(), 64))
    ga2 = GeometryAxes(shape=(64, 64))
    map1, _ = ga1.unify_with(ga2)
    assert list(map1.values())[0].size == 64


def test_feature_axes_with_variable():
    v = Variable("input", 100)
    fa = FeatureAxes(variable=v)
    assert fa.shape[0].size == 100
    assert fa.variables == v


def test_feature_axes_get_slice():
    v = Variable.from_dict({"a": 5, "b": 5})
    fa = FeatureAxes(variable=v)
    slc = fa.get_variable_slice({"a": 5})
    assert slc == [0, 1, 2, 3, 4]


def test_ellipsis_axes():
    ea = EllipsisAxes()
    assert isinstance(ea.shape[0], EllipsisDim)
    assert str(ea) == "..."


# --- Update Axes Logic ---


def test_update_axes_simple():
    d1 = AxesDim(None)
    axes = Axes(d1)
    new_d = AxesDim(50)
    changed = axes.update_axes({d1: new_d})
    assert changed is True
    assert d1.size == 50


def test_update_axes_replace_ellipsis():
    ed = EllipsisDim()
    axes = Axes(10, ed)
    new_dims = (AxesDim(20), AxesDim(30))
    changed = axes.update_axes({ed: new_dims})

    assert changed is True
    assert len(axes.shape) == 3
    assert axes.shape[1].size == 20
    assert axes.shape[2].size == 30


def test_update_axes_no_change():
    d1 = AxesDim(10)
    axes = Axes(d1)
    changed = axes.update_axes({d1: AxesDim(10)})
    assert changed is False


# --- Geometry Logic ---


def test_geometry_unify():
    g1 = Geometry(shape=(None, 32))
    g2 = Geometry(shape=(32, 32))
    g3 = g1.unify_with(g2)
    assert g3.shape[0] == 32
    assert g3.shape[1] == 32


# --- Edge Cases ---


def test_match_remainder_no_overlap():
    # Case where _match_remainder finds no overlap and creates a new ellipsis
    # This happens via internal _match_middle_shape logic
    a1 = Axes(1, ..., 2)
    a2 = Axes(1, 3, 4, 2)
    # Should unify middle part
    m1, _ = a1.unify_with(a2)
    ell_dim = a1.shape[1]
    assert len(m1[ell_dim]) == 2  # gets 3 and 4


def test_axes_dim_none_broadcastable():
    d1 = AxesDim(1, broadcastable=False)
    d2 = AxesDim(5, broadcastable=True)
    # Since d1 is not broadcastable, it cannot be expanded to 5
    with pytest.raises(DataConfigMismatchError):
        d1.unify_with(d2)


def test_axes_unify_mismatch_remaining_middle_raises():
    # Shapes that don't match and don't have ellipsis in the middle
    a1 = Axes(1, 2)
    a2 = Axes(1, 3, 2)
    with pytest.raises(DataConfigMismatchError):
        a1.unify_with(a2)


def test_axes_dim_update_runtime_error():
    axes = Axes(10)
    with pytest.raises(RuntimeError):
        axes.update_axes({axes.shape[0]: "not a dim"})  # type: ignore
