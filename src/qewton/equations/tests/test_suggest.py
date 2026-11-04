import numpy as np

from qewton.config.variables import Variable
from qewton.equations import (
    Boundary,
    Domain,
    Eq,
    FunctionSymbol,
    Problem,
    Restrict,
    div,
    fixtures,
    grad,
    report,
    run_pipeline,
    suggest,
)
from qewton.equations.analysis.suggest import RULES, Suggestion


def task_of(name, *unknown):
    problem = getattr(fixtures, name)()
    symbols = [s for s in problem.symbols() if s.variable.name in unknown]
    return problem.task(unknown=symbols)


def methods(task):
    return [s.method for s in suggest(task)]


def has(task, text):
    return any(text in m for m in methods(task))


def test_each_pde_type_gets_its_method():
    assert has(task_of("poisson", "u"), "Lagrange FEM")
    assert has(task_of("heat", "w"), "method of lines")
    assert has(task_of("advection", "u"), "upwind")
    assert has(task_of("wave", "w"), "Newmark")
    assert has(task_of("stokes", "u", "p"), "Taylor-Hood")
    assert has(task_of("delay", "u"), "ODE integrator")


def test_obstacle_gets_an_active_set_method_with_its_relation_as_evidence():
    task = task_of("obstacle", "u")
    (found,) = [s for s in suggest(task) if s.rule == "inequality"]
    assert "semismooth" in found.method
    assert [f.rule for f in found.evidence] == ["relation"]


def test_suggestions_carry_the_facts_they_come_from():
    (found,) = [s for s in suggest(task_of("heat", "w")) if s.rule == "parabolic"]
    assert found.evidence[0].rule == "block_type"
    assert "parabolic" in found.why
    assert str(found).startswith(found.method)


def test_a_nonlinear_operator_suggests_newton():
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    u = FunctionSymbol(Variable("u", dim=1), x)
    g = FunctionSymbol(Variable("g", dim=1), x)
    problem = Problem(
        (
            Eq(-div(u * grad(u, x), x), 1).on(omega),
            Eq(Restrict(u, Boundary(omega)), g).on(Boundary(omega)),
        )
    )
    assert has(problem.task(unknown=[u]), "Newton")
    assert not has(task_of("poisson", "u"), "Newton")


def test_a_problem_without_time_derivatives_has_no_time_rule():
    assert not has(task_of("poisson", "u"), "stationary solves")


def test_algebraic_equation_is_a_pointwise_solve():
    x = Variable("x", dim=1)
    omega = Domain("Omega", ambient=x)
    u = FunctionSymbol(Variable("u", dim=1), x)
    f = FunctionSymbol(Variable("f", dim=1), x)
    problem = Problem((Eq(u, f).on(omega),))
    assert has(problem.task(unknown=[u]), "pointwise solve")


def test_batched_source_suggests_a_single_factorization():
    problem = fixtures.poisson()
    by_name = {s.variable.name: s for s in problem.symbols()}
    fixed = {by_name[k]: 0.0 for k in ("g", "h")}
    fixed[by_name["nu"]] = 1.0
    task = problem.task(
        unknown=[by_name["u"]],
        fixed=fixed,
        varying={by_name["f"]: np.ones(3)},
    )
    found = [s for s in suggest(task) if s.rule == "batch_source_only"]
    assert len(found) == 1
    assert "right-hand sides" in found[0].method
    assert found[0].evidence[0].rule == "batch_mode"


def test_batched_operator_suggests_reduced_basis_or_eim():
    problem = fixtures.poisson()
    by_name = {s.variable.name: s for s in problem.symbols()}
    fixed = {by_name[k]: 0.0 for k in ("g", "h")}
    fixed[by_name["f"]] = 1.0
    task = problem.task(
        unknown=[by_name["u"]], fixed=fixed, varying={by_name["nu"]: np.ones(3)}
    )
    rules = {s.rule for s in suggest(task)}
    assert rules & {"batch_operator_affine", "batch_operator_nonlinear"}


def test_report_lists_classification_and_suggestions():
    text = report(run_pipeline(task_of("heat", "w")))
    assert "Classification" in text
    assert "Suggestions" in text
    assert "method of lines" in text


def test_report_skips_the_analysis_when_the_task_is_invalid_or_has_no_unknown():
    problem = fixtures.poisson()
    text = report(run_pipeline(problem.task()))
    assert "Suggestions" not in text


def test_every_rule_yields_suggestions():
    assert all(isinstance(s, Suggestion) for s in suggest(task_of("stokes", "u", "p")))
    assert len(RULES) >= 10
