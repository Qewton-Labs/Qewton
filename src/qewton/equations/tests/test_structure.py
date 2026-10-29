import pytest

from qewton.equations import (
    FunctionSymbol,
    Variable,
    d,
    dt,
    div,
    exp,
    grad,
    lap,
    shift,
)
from qewton.equations.analysis.structure import (
    constraint_orders,
    derivative_orders,
    linearity,
    split,
    unknown_orders,
)
from qewton.equations.fixtures import delay, heat, poisson

x = Variable("x", dim=2)
t = Variable("t", dim=1)
u = FunctionSymbol(Variable("u", dim=1), x * t)
v = FunctionSymbol(Variable("v", dim=1), x * t)


def test_a_function_without_derivatives_has_no_orders():
    assert derivative_orders(u + 1) == {u: {}}


def test_a_first_derivative_counts_once_along_its_axis():
    assert derivative_orders(dt(u, t)) == {u: {t: 1}}
    assert derivative_orders(grad(u, x)) == {u: {x: 1}}


def test_orders_add_up_along_the_path():
    assert derivative_orders(d(dt(u, t), t)) == {u: {t: 2}}
    assert derivative_orders(dt(div(grad(u, x), x), t)) == {u: {x: 2, t: 1}}


def test_lap_is_order_two_and_gradient_over_a_product_hits_every_factor():
    assert derivative_orders(lap(u, x)) == {u: {x: 2}}
    assert derivative_orders(grad(u, x * t)) == {u: {x: 1, t: 1}}


def test_occurrences_keep_their_highest_order():
    assert derivative_orders(d(u, t) + lap(u, x) + u * u) == {u: {t: 1, x: 2}}


def test_each_function_has_its_own_orders():
    assert derivative_orders(dt(u, t) * grad(v, x)) == {u: {t: 1}, v: {x: 1}}


def test_derivative_of_a_product_applies_to_both_functions():
    assert derivative_orders(dt(u * v, t)) == {u: {t: 1}, v: {t: 1}}


def test_a_shift_is_not_a_derivative():
    assert derivative_orders(shift(u, t, -1)) == {u: {}}
    assert derivative_orders(dt(shift(u, t, -1), t)) == {u: {t: 1}}


def test_constraint_orders_merge_both_sides():
    p = heat()
    first = constraint_orders(p.constraints[0])
    w = next(s for s in first if s.variable.name == "w")
    assert first[w] == {Variable("t", dim=1): 1, Variable("x", dim=2): 2}


def test_unknown_orders_for_poisson():
    p = poisson()
    u_ = next(
        s for s in p.symbols() if getattr(s, "variable", None) and s.variable.name == "u"
    )
    task = p.task(unknown=[u_], fixed={s: 1.0 for s in p.symbols() if s is not u_})
    first, second, third = unknown_orders(task)
    assert first == {u_: {Variable("x", dim=2): 2}}
    assert second == {u_: {}}
    assert third == {u_: {Variable("x", dim=2): 1}}


def test_unknown_orders_leave_out_other_functions_and_empty_constraints():
    p = delay()
    names = {s.variable.name: s for s in p.symbols()}
    task = p.task(unknown=[names["u"]], fixed={names["phi"]: 1.0})
    first, second = unknown_orders(task)
    assert first == {names["u"]: {Variable("t", dim=1): 1}}
    assert second == {names["u"]: {}}


U = frozenset({u})
nu = FunctionSymbol(Variable("nu", dim=1), None)
f = FunctionSymbol(Variable("f", dim=1), x)


@pytest.mark.parametrize(
    "expr, expected",
    [
        (f, "none"),
        (f * nu + 1, "none"),
        (u, "linear"),
        (-u, "linear"),
        (dt(u, t) - lap(u, x), "linear"),
        (nu * grad(u, x), "linear"),
        (div(nu * grad(u, x), x), "linear"),
        (u + f, "affine"),
        (u + 1, "affine"),
        (2 * (u + 1), "affine"),
        (u + u, "linear"),
        (u * u, "nonlinear"),
        (u * dt(u, t), "nonlinear"),
        (exp(u), "nonlinear"),
        (exp(f), "none"),
        (f * exp(u) + u, "nonlinear"),
    ],
)
def test_linearity_in_the_unknown(expr, expected):
    assert linearity(expr, U) == expected


def test_a_function_that_is_not_unknown_counts_as_data():
    assert linearity(u * f, U) == "linear"
    assert linearity(u * f, frozenset({u, f})) == "nonlinear"


def test_a_shift_keeps_the_dependence_and_an_unknown_offset_breaks_it():
    assert linearity(shift(u, t, -1), U) == "linear"
    tau = FunctionSymbol(Variable("tau", dim=1), None)
    g = FunctionSymbol(Variable("g", dim=1), t)
    assert linearity(shift(g, t, tau), U) == "none"
    assert linearity(shift(g, t, tau), frozenset({tau})) == "nonlinear"


def poisson_task():
    p = poisson()
    by_name = {s.variable.name: s for s in p.symbols()}
    fixed = {by_name[n]: 1.0 for n in ("f", "g", "h", "nu")}
    return p.task(unknown=[by_name["u"]], fixed=fixed)


def test_poisson_splits_into_an_operator_and_a_source():
    first, second, third = split(poisson_task())
    assert [t.sign for t in first.operator] == [-1]
    assert [(t.sign, t.expr.variable.name) for t in first.source] == [(-1, "f")]
    assert first.linearity == "linear"
    assert second.linearity == "linear"
    assert [(t.sign, t.expr.variable.name) for t in second.source] == [(-1, "g")]
    assert third.linearity == "linear"


def test_signs_follow_subtraction_and_negation():
    p = heat()
    by_name = {s.variable.name: s for s in p.symbols()}
    task = p.task(
        unknown=[by_name["w"]],
        fixed={by_name["f"]: 1.0},
        varying={by_name["w0"]: [1.0]},
    )
    first, second = split(task)
    assert [t.sign for t in first.operator] == [1, -1]
    assert [t.sign for t in first.source] == [-1]
    assert [t.sign for t in second.operator] == [1]
    assert [t.sign for t in second.source] == [-1]


def test_a_constraint_without_an_unknown_has_no_operator():
    p = poisson()
    task = p.task(unknown=[], fixed={s: 1.0 for s in p.symbols()})
    assert all(s.linearity == "none" and not s.operator for s in split(task))


def test_a_nonlinear_term_makes_the_constraint_nonlinear():
    from qewton.equations import Domain, Eq, Problem

    omega = Domain("Omega", ambient=x)
    q = FunctionSymbol(Variable("q", dim=1), x)
    problem = Problem(constraints=(Eq(-lap(q, x) + q * q, f).on(omega),))
    (result,) = split(problem.task(unknown=[q], fixed={f: 1.0}))
    assert result.linearity == "nonlinear"
    assert [t.linearity for t in result.operator] == ["linear", "nonlinear"]
    assert [t.linearity for t in result.source] == ["none"]
