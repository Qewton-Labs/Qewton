import numpy as np

from qewton.equations import (
    Domain,
    Eq,
    FunctionSymbol,
    Integral,
    Problem,
    Variable,
    signatures,
)
from qewton.equations.analysis.structure.batch_axes import inherited_batch_axes
from qewton.equations.fixtures import heat, poisson


def _named(problem, name):
    return next(s for s in problem.symbols() if s.variable.name == name)


def test_varying_initial_data_propagates_its_axis_to_the_unknown():
    problem = heat()
    w, w0 = _named(problem, "w"), _named(problem, "w0")
    omega = problem.constraints[0].on.factors[0]
    task = problem.task(
        unknown=[w], varying={w0: np.ones((4, 20))}, fixed={omega: object()}
    )
    axes = inherited_batch_axes(task)

    assert axes == {w: (task.batch_axes[w0],)}
    signature = signatures(task)[w]
    assert signature.inherited_batch == (task.batch_axes[w0],)
    assert "w : x × t × p_w0 →" in str(signature)
    assert "[unknown; inherits p_w0]" in str(signature)


def test_batches_propagate_transitively_but_not_to_decoupled_unknowns():
    x = Variable("x", dim=1)
    domain = Domain("Omega", ambient=x)
    u = FunctionSymbol(Variable("u", dim=1), x)
    v = FunctionSymbol(Variable("v", dim=1), x)
    w = FunctionSymbol(Variable("w", dim=1), x)
    varying = FunctionSymbol(Variable("f", dim=1), x)
    fixed = FunctionSymbol(Variable("g", dim=1), x)
    problem = Problem(
        (
            Eq(u, varying).on(domain),
            Eq(u, v).on(domain),
            Eq(w, fixed).on(domain),
        )
    )
    task = problem.task(
        unknown=[u, v, w],
        varying={varying: np.ones(3)},
        fixed={fixed: np.ones(3), domain: object()},
    )

    assert set(inherited_batch_axes(task)) == {u, v}
    assert inherited_batch_axes(task)[u] == inherited_batch_axes(task)[v]


def test_varying_domain_batch_reaches_unknown_on_its_boundary():
    problem = poisson()
    unknown = _named(problem, "u")
    omega = next(s for s in problem.sets() if str(s) == "Omega")
    task = problem.task(
        unknown=[unknown],
        fixed={symbol: 1.0 for symbol in problem.symbols() if symbol is not unknown},
        varying={omega: object()},
    )

    assert inherited_batch_axes(task)[unknown] == (task.batch_axes[omega],)


def test_moving_partition_part_propagates_to_sibling_constraint():
    problem = poisson()
    unknown = _named(problem, "u")
    gamma_d = next(s for s in problem.sets() if str(s) == "Gamma_D")
    task = problem.task(
        unknown=[unknown],
        fixed={symbol: 1.0 for symbol in problem.symbols() if symbol is not unknown},
        varying={gamma_d: "moving split"},
    )

    assert inherited_batch_axes(task)[unknown] == (task.batch_axes[gamma_d],)


def test_varying_set_inside_a_global_integral_seeds_the_batch_axis():
    problem = poisson()
    unknown = _named(problem, "u")
    gamma_d = next(s for s in problem.sets() if str(s) == "Gamma_D")
    integral_problem = Problem(
        (Eq(Integral(unknown, gamma_d), 0).on(None),),
        partitions=problem.partitions,
    )
    task = integral_problem.task(unknown=[unknown], varying={gamma_d: "moving"})

    assert inherited_batch_axes(task)[unknown] == (task.batch_axes[gamma_d],)


def test_multiple_varying_inputs_are_all_inherited_in_stable_order():
    problem = heat()
    w, w0 = _named(problem, "w"), _named(problem, "w0")
    # Use two varied functions in the heat constraints, each of which connects to w.
    # The fixture's source is f, which is constant, and can be marked varying too.
    f = _named(problem, "f")
    task = problem.task(
        unknown=[w],
        varying={w0: np.ones((2, 20)), f: np.ones(2)},
        fixed={problem.constraints[0].on.factors[0]: object()},
    )

    axes = inherited_batch_axes(task)[w]
    assert axes == tuple(
        sorted((task.batch_axes[w0], task.batch_axes[f]), key=lambda axis: axis.name)
    )


def test_a_fixed_set_does_not_inherit_an_unrelated_moving_set_axis():
    x = Variable("x", dim=1)
    omega = Domain("Omega", ambient=x)
    other = Domain("Other", ambient=x)
    u = FunctionSymbol(Variable("u", dim=1), x)
    v = FunctionSymbol(Variable("v", dim=1), x)
    f = FunctionSymbol(Variable("f", dim=1), x)
    g = FunctionSymbol(Variable("g", dim=1), x)
    problem = Problem((Eq(u, f).on(omega), Eq(v, g).on(other)))
    task = problem.task(
        unknown=[u, v],
        varying={other: "moving"},
        fixed={f: 1.0, g: 1.0},
    )

    assert u not in inherited_batch_axes(task)
    assert inherited_batch_axes(task)[v] == (task.batch_axes[other],)
