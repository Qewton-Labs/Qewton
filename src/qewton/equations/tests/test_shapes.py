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

x = Variable("x", dim=2)
t = Variable("t", dim=1)
u = FunctionSymbol(Variable("u", dim=1), x)
v = FunctionSymbol(Variable("v", dim=2), x)
u_xt = FunctionSymbol(Variable("u", dim=1), x * t)
K = FunctionSymbol(Variable("K", dim=(2, 2)), None)


def test_leaves():
    assert u.shape == ()
    assert v.shape == (2,)
    assert K.shape == (2, 2)
    assert Const([[1, 2], [3, 4]]).shape == (2, 2)


def test_derivatives():
    assert grad(u, x).shape == (2,)
    assert grad(v, x).shape == (2, 2)
    assert div(v, x).shape == ()
    assert lap(u, x).shape == ()
    assert grad(u_xt, x * t).shape == (3,)


def test_products():
    assert dot(K, v).shape == (2,)
    assert (u * v).shape == (2,)
    assert div(dot(K, grad(u, x)), x).shape == ()


def test_normal_and_restrict():
    omega = Domain("Omega", ambient=x)
    n = Normal(Boundary(omega))
    assert n.shape == (2,)
    assert dot(Restrict(grad(u, x), Boundary(omega)), n).shape == ()


def test_mismatch_names_the_expression():
    with pytest.raises(ShapeError, match="div"):
        div(u, x).shape
    with pytest.raises(ShapeError):
        (v + K).shape
    with pytest.raises(ShapeError):
        (v * v).shape


def test_undeclared_output_is_unknown():
    q = FunctionSymbol(Variable("q"), x)
    assert q.shape is None
    assert (grad(q, x) + 1).shape is None


def test_apply_checks_at_construction():
    with pytest.raises(ShapeError, match="div"):
        div(u, x)
    with pytest.raises(ShapeError):
        v * v


def test_apply_skips_undeclared_shapes():
    q = FunctionSymbol(Variable("q"), x)
    expr = div(q, x)  # builds: q's shape is not known yet
    assert expr is not None
    assert expr.shape is None


def test_constraint_checks_both_sides():
    omega = Domain("Omega", ambient=x)
    with pytest.raises(ShapeError, match="sides"):
        Eq(grad(u, x), u).on(omega)
    Eq(grad(u, x), grad(u, x)).on(omega)
    Eq(v, 0).on(omega)  # a scalar constant stands for any shape


def test_constraint_skips_undeclared_shapes():
    q = FunctionSymbol(Variable("q"), x)
    Eq(grad(u, x), q).on(Domain("Omega", ambient=x))
