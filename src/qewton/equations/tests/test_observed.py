import numpy as np
import pytest

from qewton.equations import (
    Domain,
    Eq,
    FunctionSymbol,
    Observed,
    Problem,
    Slice,
    TaskError,
    TrainingMode,
    Variable,
    dt,
    integral,
    lap,
    report,
    run_pipeline,
    suggest,
    training_modes,
)


def heat_problem():
    x, t = Variable("x", dim=2), Variable("t", dim=1)
    u = FunctionSymbol(Variable("u", dim=1), x * t)
    nu = FunctionSymbol(Variable("nu", dim=1), x)
    omega, interval = Domain("Omega", ambient=x), Domain("I", ambient=t)
    bulk, start = omega * interval, omega * Slice(t, 0.0)
    problem = Problem(
        (Eq(dt(u, t) - nu * lap(u, x), 0).on(bulk), Eq(u, 0).on(start)), time=t
    )
    return problem, u, nu, bulk, start


def test_a_task_without_samples_is_physics_informed():
    problem, u, *_ = heat_problem()
    task = problem.task(unknown=[u])
    assert task.observed == {}
    assert training_modes(task) == {u: TrainingMode.PHYSICS_INFORMED}


def test_samples_everywhere_or_on_the_bulk_are_data_driven():
    problem, u, _, bulk, _ = heat_problem()
    for data in (Observed(), Observed(on=bulk, samples=500)):
        task = problem.task(unknown=[u], observed={u: data})
        assert training_modes(task)[u] is TrainingMode.DATA_DRIVEN


def test_samples_on_the_initial_slice_only_are_hybrid():
    problem, u, _, _, start = heat_problem()
    task = problem.task(unknown=[u], observed={u: Observed(on=start)})
    assert training_modes(task)[u] is TrainingMode.HYBRID


def test_the_task_prints_its_observations():
    problem, u, *_ = heat_problem()
    task = problem.task(unknown=[u], observed={u: Observed(samples=10, noisy=True)})
    assert "observed u: everywhere, 10 samples, noisy" in str(task)


def test_only_unknowns_can_be_observed():
    problem, u, nu, *_ = heat_problem()
    with pytest.raises(TaskError, match="not an unknown"):
        problem.task(unknown=[u], fixed=[nu], observed={nu: Observed()})


def test_samples_must_lie_on_the_inputs_of_the_function():
    problem, u, *_ = heat_problem()
    s = Variable("s", dim=1)
    with pytest.raises(TaskError, match="not among the function's inputs"):
        problem.task(unknown=[u], observed={u: Observed(on=Domain("S", ambient=s))})


def test_observed_validates_its_arguments():
    with pytest.raises(ValueError):
        Observed(samples=0)
    with pytest.raises(TypeError):
        Observed(on="Omega")  # type: ignore[arg-type]


def test_a_plain_forward_task_gets_no_learning_suggestion():
    problem, u, *_ = heat_problem()
    task = problem.task(unknown=[u])
    assert not [s for s in suggest(task) if s.rule.startswith("learning")]


def test_operator_learning_suggestions_follow_the_data():
    problem, u, nu, bulk, _ = heat_problem()
    varying = {nu: np.ones((4,))}
    pinn = problem.task(unknown=[u], varying=varying)
    (found,) = [s for s in suggest(pinn) if s.rule.startswith("learning")]
    assert found.rule == "learning_physics_informed"
    assert "varying inputs" in found.method

    data = Observed(on=bulk, samples=100, noisy=True)
    supervised = problem.task(unknown=[u], varying=varying, observed={u: data})
    (found,) = [s for s in suggest(supervised) if s.rule.startswith("learning")]
    assert found.rule == "learning_data_driven"
    assert "operator learning" in found.method
    assert "noisy" in found.why
    assert found.evidence[0].rule == "training_mode"


def test_observed_unknown_without_batch_gets_a_surrogate_and_shows_in_the_report():
    problem, u, _, bulk, _ = heat_problem()
    task = problem.task(unknown=[u], observed={u: Observed(on=bulk, samples=50)})
    (found,) = [s for s in suggest(task) if s.rule.startswith("learning")]
    assert "regression" in found.method
    assert "observed u" in report(run_pipeline(task))


def test_a_kernel_integral_over_a_copy_of_the_interval_passes_the_pipeline():
    x, y = Variable("x", dim=1), Variable("y", dim=1)
    kernel = FunctionSymbol(Variable("K", dim=1), x * y)
    f = FunctionSymbol(Variable("f", dim=1), y)
    g = FunctionSymbol(Variable("g", dim=1), x)
    constraint = Eq(g, integral(kernel * f, Domain("Iy", ambient=y))).on(
        Domain("I", ambient=x)
    )
    task = Problem((constraint,)).task(unknown=[g], fixed=[kernel, f])
    assert run_pipeline(task).ok
