import pytest

from qewton.equations import (
    Boundary,
    Const,
    Eq,
    Domain,
    FunctionSymbol,
    Normal,
    Restrict,
    Variable,
    div,
    dot,
    grad,
    lap,
)
from qewton.equations.expr.ops import ShapeError
from qewton.equations.analysis.shapes import UnknownShapeError, shape_of

x = Variable("x", dim=2)
t = Variable("t", dim=1)
u = FunctionSymbol(Variable("u", dim=1), x)
v = FunctionSymbol(Variable("v", dim=2), x)
u_xt = FunctionSymbol(Variable("u", dim=1), x * t)
K = FunctionSymbol(Variable("K", dim=(2, 2)), None)


def test_leaves():
    assert shape_of(u) == ()
    assert shape_of(v) == (2,)
    assert shape_of(K) == (2, 2)
    assert shape_of(Const([[1, 2], [3, 4]])) == (2, 2)


def test_derivatives():
    assert shape_of(grad(u, x)) == (2,)
    assert shape_of(grad(v, x)) == (2, 2)
    assert shape_of(div(v, x)) == ()
    assert shape_of(lap(u, x)) == ()
    assert shape_of(grad(u_xt, x * t)) == (3,)


def test_products():
    assert shape_of(dot(K, v)) == (2,)
    assert shape_of(u * v) == (2,)
    assert shape_of(div(dot(K, grad(u, x)), x)) == ()


def test_normal_and_restrict():
    omega = Domain("Omega", ambient=x)
    n = Normal(Boundary(omega))
    assert shape_of(n) == (2,)
    assert shape_of(dot(Restrict(grad(u, x), Boundary(omega)), n)) == ()


def test_mismatch_names_the_expression():
    with pytest.raises(ShapeError, match="div"):
        shape_of(div(u, x))
    with pytest.raises(ShapeError):
        shape_of(v + K)
    with pytest.raises(ShapeError):
        shape_of(v * v)


def test_undeclared_output_is_unknown():
    q = FunctionSymbol(Variable("q"), x)
    with pytest.raises(UnknownShapeError):
        shape_of(q)
    with pytest.raises(UnknownShapeError):
        shape_of(grad(q, x) + 1)


def test_apply_checks_at_construction():
    with pytest.raises(ShapeError, match="div"):
        div(u, x)
    with pytest.raises(ShapeError):
        v * v


def test_apply_skips_undeclared_shapes():
    q = FunctionSymbol(Variable("q"), x)
    expr = div(q, x)  # builds: q's shape is not known yet
    assert expr is not None
    with pytest.raises(UnknownShapeError):
        shape_of(expr)


def test_constraint_checks_both_sides():
    omega = Domain("Omega", ambient=x)
    with pytest.raises(ShapeError, match="sides"):
        Eq(grad(u, x), u).on(omega)
    Eq(grad(u, x), grad(u, x)).on(omega)
    Eq(v, 0).on(omega)  # a scalar constant stands for any shape


def test_constraint_skips_undeclared_shapes():
    q = FunctionSymbol(Variable("q"), x)
    Eq(grad(u, x), q).on(Domain("Omega", ambient=x))
