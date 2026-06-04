import pytest
from qewton.config.variables import Variable


def test_variable_init_default():
    v = Variable()
    assert v.is_empty()


def test_variable_init_single():
    v = Variable("x", 5)
    assert not v.is_empty()
    assert "x" in v
    assert v["x"] == 5
    assert v.dim == 5
    assert v.shape == (5,)


def test_variable_init_tuple_dim():
    v = Variable("img", (3, 32, 32))
    assert v.has_multiple_axes
    assert v.dim == 3 * 32 * 32
    assert v.shape == (3, 32, 32)


def test_from_dict():
    d = {"a": 2, "b": 3}
    v = Variable.from_dict(d)
    assert v["a"] == 2
    assert v["b"] == 3
    assert v.dim == 5
    assert list(v.keys()) == ["a", "b"]


def test_is_empty():
    assert Variable().is_empty()
    assert not Variable("x", 1).is_empty()
    # A variable initialized with parameters is not empty even if dim is None
    assert not Variable("x", None).is_empty()


def test_dim_calculation_mixed():
    v = Variable.from_dict({"a": 10, "b": 20})
    assert v.dim == 30


def test_shape_property():
    v1 = Variable("x", 10)
    assert v1.shape == (10,)

    v2 = Variable("x", (2, 5))
    assert v2.shape == (2, 5)


def test_add_variables():
    v1 = Variable("a", 2)
    v2 = Variable("b", 3)
    v3 = v1 + v2
    assert list(v3.keys()) == ["a", "b"]
    assert v3.dim == 5
    assert v3["a"] == 2
    assert v3["b"] == 3


def test_add_overlapping_names_raises():
    v1 = Variable("a", 2)
    v2 = Variable("a", 3)
    with pytest.raises(ValueError, match="overlapping names"):
        _ = v1 + v2


def test_add_multiple_axes_raises():
    v1 = Variable("a", (2, 2))
    v2 = Variable("b", 3)
    with pytest.raises(ValueError, match="multiple axes"):
        _ = v1 + v2


def test_unify_empty():
    v_empty = Variable()
    v = Variable("x", 10)
    assert v_empty.unify(v) == v
    assert v.unify(v_empty) == v


def test_unify_identical():
    v1 = Variable("x", 10)
    v2 = Variable("x", 10)
    v3 = v1.unify(v2)
    assert v3["x"] == 10
    assert len(v3) == 1


def test_unify_dim_propagation():
    v1 = Variable("x", None)
    v2 = Variable("x", 10)
    v3 = v1.unify(v2)
    assert v3["x"] == 10


def test_unify_mismatch_names_raises():
    v1 = Variable("x", 10)
    v2 = Variable("y", 10)
    with pytest.raises(ValueError, match="names have to agree"):
        v1.unify(v2)


def test_unify_mismatch_dims_raises():
    v1 = Variable("x", 10)
    v2 = Variable("x", 20)
    with pytest.raises(ValueError, match="dimensions have to agree"):
        v1.unify(v2)


def test_get_slice_simple():
    v = Variable.from_dict({"a": 5, "b": 10})
    # Slice for 'a' starts at 0
    assert v.get_slice({"a": 5}) == [0, 1, 2, 3, 4]
    # Slice for 'b' starts at 5
    assert v.get_slice({"b": 10}) == list(range(5, 15))


def test_get_slice_multi_axes():
    v = Variable("x", (2, 3, 4))
    # Multiple axes variables return full slices for all axes
    assert v.get_slice({"x": (2, 3, 4)}) == (slice(None), slice(None), slice(None))


def test_contains_str():
    v = Variable.from_dict({"a": 5, "b": 10})
    assert "a" in v
    assert "c" not in v


def test_contains_variable():
    v = Variable.from_dict({"a": 5, "b": 10})
    assert Variable("a", 5) in v
    assert Variable("a", 6) not in v
    assert Variable("c", 5) not in v


def test_contains_invalid_type():
    v = Variable("x", 5)
    assert 123 not in v


def test_getitem_single_name():
    v = Variable.from_dict({"a": 5, "b": 10})
    assert v["a"] == 5


def test_getitem_list_of_names():
    v = Variable.from_dict({"a": 5, "b": 10, "c": 15})
    v_sub = v[["a", "c"]]
    assert isinstance(v_sub, Variable)
    assert list(v_sub.keys()) == ["a", "c"]
    assert v_sub["a"] == 5
    assert v_sub["c"] == 15


def test_getitem_slice_of_names():
    v = Variable.from_dict({"a": 1, "b": 2, "c": 3, "d": 4})
    # Slicing from 'b' to 'd' (exclusive of 'd')
    v_sub = v["b":"d"]
    assert list(v_sub.keys()) == ["b", "c"]


def test_getitem_slice_open_ended():
    v = Variable.from_dict({"a": 1, "b": 2, "c": 3})
    assert list(v[:"b"].keys()) == ["a"]
    assert list(v["b":].keys()) == ["b", "c"]
    assert list(v[:].keys()) == ["a", "b", "c"]


def test_hash_logic():
    v1 = Variable.from_dict({"a": 5, "b": 10})
    v2 = Variable.from_dict({"a": 5, "b": 10})
    v3 = Variable.from_dict({"b": 10, "a": 5})
    assert hash(v1) == hash(v2)
    # Order matters for hash string construction in this implementation
    assert hash(v1) != hash(v3)


def test_repr_output():
    v = Variable("x", 5)
    assert repr(v) == "Variable({'x': 5})"


def test_mul_operator_behavior():
    v1 = Variable("a", 1)
    v2 = Variable("b", 2)
    v3 = v1 * v2
    assert v3.dim == 3
    assert list(v3.keys()) == ["a", "b"]


def test_check_static_helper():
    v1 = Variable("x", 10)
    v2 = Variable("x", 10)
    v_none = Variable("x", None)

    assert Variable.check(v1, v2, "x", "x") == 10
    assert Variable.check(v_none, v1, "x", "x") == 10
    assert Variable.check(v1, v_none, "x", "x") == 10

    with pytest.raises(ValueError, match="dimensions have to agree"):
        Variable.check(v1, Variable("x", 5), "x", "x")
