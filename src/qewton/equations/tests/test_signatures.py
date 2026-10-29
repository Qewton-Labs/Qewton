import numpy as np
import pytest

from qewton.equations import (
    Boundary,
    Domain,
    Eq,
    FunctionSymbol,
    Problem,
    Product,
    Restrict,
    Role,
    Severity,
    Slice,
    Variable,
    dt,
    integral,
)
from qewton.equations.analysis.signatures import signatures
from qewton.equations.fixtures import delay, heat, periodic_poisson, poisson

x = Variable("x", dim=2)
t = Variable("t", dim=1)
Omega = Domain("Omega", ambient=x)
I = Domain("I", ambient=t)


def sig(problem, name, task=None):
    result = signatures(task or problem.task())
    symbol = next(s for s in result.table if s.variable.name == name)
    return result[symbol]


def test_every_function_has_a_signature():
    p = poisson()
    result = signatures(p.task())
    assert set(result.table) == p.symbols()
    assert len(list(result)) == 5


def test_axes_come_in_the_declared_order():
    s = sig(heat(), "w")
    assert [a.name for a in s.axes] == ["x", "t"]


def test_constants_have_no_axes():
    assert sig(poisson(), "nu").axes == ()


def test_shapes_come_from_inference():
    s = sig(poisson(), "u")
    assert s.shape == () and not s.shape_assumed
    assert sig(heat(), "w").shape_assumed


def test_roles_come_from_the_task():
    p = poisson()
    u = next(s for s in p.symbols() if s.variable.name == "u")
    assert sig(p, "u", p.task(unknown=[u])).role is Role.UNKNOWN
    assert sig(p, "nu", p.task(unknown=[u])).role is Role.FIXED


def test_varying_functions_carry_their_batch_axis():
    p = poisson()
    nu = next(s for s in p.symbols() if s.variable.name == "nu")
    task = p.task(varying={nu: np.ones(4)})
    s = sig(p, "nu", task)
    assert s.batch.name == "p_nu"
    assert "nu : p_nu → ℝ" in str(s)
    assert sig(p, "u", task).batch is None


def test_the_sets_are_those_where_the_function_is_evaluated():
    s = sig(poisson(), "u")
    assert {str(o) for o in s.evaluated_on} == {"Omega", "Gamma_D", "Gamma_N"}
    assert {str(o) for o in sig(poisson(), "g").evaluated_on} == {"Gamma_D"}


def test_sets_inside_another_evaluation_set_are_dropped():
    s = sig(periodic_poisson(), "f")
    assert [str(o) for o in s.evaluated_on] == ["Omega"]


def test_time_lifting_is_applied_first():
    q = FunctionSymbol(Variable("q", dim=1), x * t)
    dO = Boundary(Omega)
    problem = Problem(
        (
            Eq(dt(q, t), 0).on(Product((Omega, I))),
            Eq(Restrict(q, dO), 0).on(dO),
        ),
        time=t,
    )
    s = sig(problem, "q")
    assert "∂Omega × I" in {str(o) for o in s.evaluated_on}


def test_an_integral_evaluates_on_its_set():
    p = FunctionSymbol(Variable("p", dim=1), x)
    problem = Problem((Eq(integral(p, Omega), 0).on(None),))
    assert [str(o) for o in sig(problem, "p").evaluated_on] == ["Omega"]


def test_delay_history_is_evaluated_on_its_interval():
    s = sig(delay(), "phi")
    assert [str(o) for o in s.evaluated_on] == ["H"]


class TestUnusedInputs:
    def test_fixtures_are_clean(self):
        for make in (poisson, heat, delay, periodic_poisson):
            assert signatures(make().task()).diagnostics == ()

    def test_an_input_no_constraint_can_use_is_warned(self):
        # No interval is declared, so nothing lifts the constraint to hold in time.
        q = FunctionSymbol(Variable("q", dim=1), x * t)
        problem = Problem((Eq(q, 0).on(Omega),), time=t)
        found = signatures(problem.task()).diagnostics
        assert [d.severity for d in found] == [Severity.WARNING]
        assert "depend on t" in found[0].message

    def test_a_slice_counts_as_using_its_axis(self):
        q = FunctionSymbol(Variable("q", dim=1), x * t)
        start = Product((Omega, Slice(t, 0.0)))
        problem = Problem((Eq(Restrict(q, start), 0).on(start),), time=t)
        assert signatures(problem.task()).diagnostics == ()


def test_the_report_prints_one_line_per_function():
    text = str(signatures(poisson().task()))
    assert text.count("\n") == 4
    assert "u : x → ℝ  [fixed]  on Omega, Gamma_D, Gamma_N" in text
