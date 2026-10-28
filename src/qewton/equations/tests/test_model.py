import pytest

from qewton.config.variables import Variable
from qewton.equations.constraints import Eq
from qewton.equations.expr import api
from qewton.equations.expr.nodes import FunctionSymbol, Normal, Restrict, walk
from qewton.equations.geometry.sets import (
    Boundary,
    Domain,
    Part,
    Partition,
    Product,
    Slice,
)
from qewton.equations.model import Problem

x = Variable("x", dim=2)
t = Variable("t", dim=1)
Omega = Domain("Omega", ambient=x)
Interval = Domain("I", ambient=t)
dO = Boundary(Omega)
GD = Part("Gamma_D", dO)
GN = Part("Gamma_N", dO)
u = FunctionSymbol(Variable("u"), x)
f = FunctionSymbol(Variable("f"), None)
g = FunctionSymbol(Variable("g"), x)
h = FunctionSymbol(Variable("h"), x)
nu = FunctionSymbol(Variable("nu"), None)


def poisson_problem():
    """The worked example of the plan, section 8."""
    return Problem(
        constraints=(
            Eq(-api.div(nu * api.grad(u, x), x), f).on(Omega),
            Eq(Restrict(u, GD), g).on(GD),
            Eq(api.dot(Restrict(nu * api.grad(u, x), GN), Normal(dO)), h).on(GN),
        ),
        partitions=(Partition(dO, (GD, GN)),),
    )


class TestProblemStructure:
    def test_needs_a_constraint(self):
        with pytest.raises(ValueError):
            Problem(())

    def test_a_pending_constraint_is_rejected_with_a_hint(self):
        with pytest.raises(TypeError, match=r"\.on\("):
            Problem((Eq(u, 0.0),))

    def test_sequences_become_tuples(self):
        problem = Problem([Eq(u, 0.0).on(Omega)], partitions=[Partition(dO, (GD, GN))])
        assert isinstance(problem.constraints, tuple)
        assert isinstance(problem.partitions, tuple)
        assert isinstance(hash(problem), int)

    def test_partitions_must_be_partitions(self):
        with pytest.raises(TypeError):
            Problem((Eq(u, 0.0).on(Omega),), partitions=(GD,))

    def test_time_must_be_one_dimensional(self):
        with pytest.raises(ValueError):
            Problem((Eq(u, 0.0).on(Omega),), time=x)
        assert Problem((Eq(u, 0.0).on(Omega),), time=t).time == t

    def test_equal_problems_are_equal(self):
        assert poisson_problem() == poisson_problem()
        assert hash(poisson_problem()) == hash(poisson_problem())


class TestDerivedQueries:
    def test_symbols_are_all_functions_in_the_constraints(self):
        assert poisson_problem().symbols() == frozenset({u, f, g, h, nu})

    def test_sets_include_constraint_sets_restrictions_normals_and_partition(self):
        assert poisson_problem().sets() == frozenset({Omega, GD, GN, dO})

    def test_time_dependent_problem(self):
        heat = Problem(
            (Eq(api.dt(u, t) - api.lap(u, x), f).on(Product((Omega, Interval))),),
            time=t,
        )
        assert heat.symbols() == frozenset({u, f})
        assert Product((Omega, Interval)) in heat.sets()

    def test_initial_condition_on_a_slice(self):
        initial = Eq(Restrict(u, Product((Omega, Slice(t, 0.0)))), g).on(
            Product((Omega, Slice(t, 0.0)))
        )
        assert g in Problem((initial,)).symbols()


class TestWalk:
    def test_walk_visits_every_sub_expression(self):
        nodes = list(walk(api.div(u, x)))
        assert nodes[0] == api.div(u, x)
        assert u in nodes

    def test_walk_of_a_leaf_is_the_leaf(self):
        assert list(walk(f)) == [f]
