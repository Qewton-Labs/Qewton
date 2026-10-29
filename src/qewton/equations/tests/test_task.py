import numpy as np
import pytest

from qewton.equations import (
    Domain,
    Eq,
    FunctionSymbol,
    Problem,
    Role,
    Severity,
    Task,
    TaskError,
    Variable,
    sin,
)
from qewton.equations.expr.nodes import Coord
from qewton.equations.fixtures import heat, poisson


def symbol(problem, name):
    return next(s for s in problem.symbols() if s.variable.name == name)


def omega(problem):
    return next(s for s in problem.sets() if str(s) == "Omega")


@pytest.fixture
def p():
    return poisson()


def test_unlisted_functions_default_to_fixed(p):
    task = p.task(unknown=[symbol(p, "u")])
    assert isinstance(task, Task)
    assert [s.variable.name for s in task.unknowns] == ["u"]
    assert {s.variable.name for s in task.with_role(Role.FIXED)} == {"f", "g", "h", "nu"}


def test_symbolic_fixed_functions_are_warned_about(p):
    task = p.task(unknown=[symbol(p, "u")], fixed={symbol(p, "nu"): 1.0})
    flagged = {d.where.variable.name for d in task.diagnostics}
    assert flagged == {"f", "g", "h"}
    assert {d.severity for d in task.diagnostics} == {Severity.WARNING}


def test_given_values_are_not_warned_about(p):
    fixed = {symbol(p, "nu"): 1.0, symbol(p, "f"): 2.0}
    task = p.task(unknown=[symbol(p, "u")], fixed=fixed)
    assert {d.where.variable.name for d in task.diagnostics} == {"g", "h"}


def test_varying_functions_get_a_parameter_axis(p):
    nu = symbol(p, "nu")
    task = p.task(unknown=[symbol(p, "u")], varying={nu: np.array([1.0, 2.0, 3.0])})
    axis = task.batch_axes[nu]
    assert axis.name == "p_nu" and axis.dim == 1
    assert task.roles[nu] is Role.VARYING


def test_batch_axis_avoids_existing_names():
    clash = Variable("p_nu", dim=1)
    nu = FunctionSymbol(Variable("nu", dim=1), clash)
    u = FunctionSymbol(Variable("u", dim=1), clash)
    omega_ = Domain("Omega", ambient=clash)
    problem = Problem((Eq(u, nu).on(omega_),))
    assert problem.task(unknown=[u], varying=[nu]).batch_axes[nu].name == "p_nu_"


def test_unknown_must_be_in_the_problem(p):
    other = symbol(heat(), "w")
    with pytest.raises(TaskError, match="not in the problem"):
        p.task(unknown=[other])


def test_a_function_has_one_role(p):
    u = symbol(p, "u")
    with pytest.raises(TaskError, match="unknown and as fixed"):
        p.task(unknown=[u], fixed=[u])


def test_unknown_has_no_value(p):
    with pytest.raises(TaskError, match="cannot be given a value"):
        p.task(unknown={symbol(p, "u"): 1.0})


def test_keys_must_be_functions_or_sets(p):
    with pytest.raises(TypeError):
        p.task(fixed={"f": 1.0})


def test_a_factor_of_a_product_can_get_a_mesh():
    h = heat()
    space = next(s for s in h.sets() if str(s).startswith("Omega"))
    factor = space.factors[0]
    task = h.task(unknown=[symbol(h, "w")], fixed={factor: "mesh"})
    assert task.values[factor] == "mesh"


def test_set_values_are_meshes(p):
    task = p.task(unknown=[symbol(p, "u")], fixed={omega(p): "mesh"})
    assert task.values[omega(p)] == "mesh"
    assert task.roles[omega(p)] is Role.FIXED


class TestValues:
    def test_constant_with_a_number(self, p):
        p.task(fixed={symbol(p, "nu"): 2.0})

    def test_constant_with_an_array_of_the_wrong_shape(self, p):
        with pytest.raises(TaskError, match="shape"):
            p.task(fixed={symbol(p, "nu"): np.ones(5)})

    def test_batched_constant_checks_the_trailing_shape(self, p):
        nu = symbol(p, "nu")
        p.task(varying={nu: np.ones(5)})
        with pytest.raises(TaskError, match="shape"):
            p.task(varying={nu: np.ones((5, 2))})

    def test_expression_may_use_the_inputs(self, p):
        f = symbol(p, "g")
        x = f.inputs
        p.task(fixed={f: sin(Coord(x, 0))})

    def test_expression_may_not_use_other_axes(self, p):
        nu = symbol(p, "nu")
        with pytest.raises(TaskError, match="not among its inputs"):
            p.task(fixed={nu: sin(Coord(symbol(p, "g").inputs, 0))})

    def test_a_single_value_for_a_function_of_space_is_a_constant(self, p):
        g = symbol(p, "g")
        p.task(fixed={g: 0.0})
        p.task(varying={g: np.ones(7)})

    def test_array_over_space_needs_a_mesh(self, p):
        g = symbol(p, "g")
        with pytest.raises(TaskError, match="give a mesh"):
            p.task(fixed={g: np.ones(10)})
        p.task(fixed={g: np.ones(10), omega(p): "mesh"})

    def test_time_axis_needs_no_mesh(self):
        h = heat()
        w = symbol(h, "w")
        assert w.inputs is not None
        task = h.task(unknown=[w])
        assert task.unknowns == (w,)


def test_an_unknown_may_be_constant():
    c = FunctionSymbol(Variable("c", dim=1), None)
    problem = Problem((Eq(c, 1).on(Domain("Omega", ambient=Variable("x", dim=1))),))
    assert problem.task(unknown=[c]).unknowns == (c,)


def test_inverse_problem_has_two_unknowns(p):
    task = p.task(unknown=[symbol(p, "u"), symbol(p, "nu")])
    assert [s.variable.name for s in task.unknowns] == ["nu", "u"]


def test_task_prints_its_roles(p):
    nu = symbol(p, "nu")
    text = str(p.task(unknown=[symbol(p, "u")], varying=[nu]))
    assert "unknown: u" in text
    assert "varying: nu" in text
    assert "batch axis of nu: p_nu" in text


class TestSets:
    def test_a_set_can_vary(self, p):
        task = p.task(unknown=[symbol(p, "u")], varying={omega(p): "meshes"})
        assert task.roles[omega(p)] is Role.VARYING
        assert task.batch_axes[omega(p)].name == "p_Omega"
        assert "varying sets: Omega" in str(task)

    def test_derived_sets_vary_with_their_base(self, p):
        task = p.task(unknown=[symbol(p, "u")], varying={omega(p): "meshes"})
        assert all(task.varies(s) for s in p.sets())

    def test_fixed_sets_do_not_vary(self, p):
        task = p.task(unknown=[symbol(p, "u")], fixed={omega(p): "mesh"})
        assert not any(task.varies(s) for s in p.sets())

    def test_a_set_cannot_be_unknown(self, p):
        with pytest.raises(TaskError, match="cannot be unknown"):
            p.task(unknown=[omega(p)])

    def test_a_moving_split_on_a_fixed_boundary(self, p):
        names = {str(s): s for s in p.sets()}
        task = p.task(unknown=[symbol(p, "u")], varying={names["Gamma_D"]: "split"})
        assert task.varies(names["Gamma_D"]) and task.varies(names["Gamma_N"])
        assert not task.varies(names["Omega"])
        assert not task.varies(names["∂Omega"])
