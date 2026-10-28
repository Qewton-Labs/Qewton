import pytest

from qewton.config.variables import Variable
from qewton.equations.constraints import (
    Complementary,
    Constraint,
    Eq,
    Ge,
    Le,
    PendingConstraint,
    Relation,
)
from qewton.equations.expr.nodes import Const, FunctionSymbol
from qewton.equations.geometry.sets import Boundary, Domain, Part

x = Variable("x", dim=2)
Omega = Domain("Omega", ambient=x)
dO = Boundary(Omega)
GD = Part("Gamma_D", dO)
u = FunctionSymbol(Variable("u"), x)


class TestEq:
    def test_builds_a_pending_constraint(self):
        assert isinstance(Eq(u, 0.0), PendingConstraint)

    def test_on_attaches_the_set(self):
        constraint = Eq(u, 0.0).on(Omega)
        assert constraint.rel is Relation.EQ
        assert constraint.on == Omega
        assert constraint.rhs == Const(0.0)
        assert constraint.name is None

    def test_name_is_optional(self):
        assert Eq(u, 0.0).on(Omega, name="pde").name == "pde"

    def test_numbers_and_vectors_become_constants(self):
        assert Eq(u, (0.0, 1.0)).on(Omega).rhs == Const((0.0, 1.0))

    def test_pending_constraint_is_not_a_constraint(self):
        assert not isinstance(Eq(u, 0.0), Constraint)

    def test_equal_constraints_are_equal(self):
        assert Eq(u, 0.0).on(Omega) == Eq(u, 0.0).on(Omega)

    def test_name_is_part_of_equality(self):
        assert Eq(u, 0.0).on(Omega, "a") != Eq(u, 0.0).on(Omega, "b")


class TestOtherRelations:
    def test_less_equal(self):
        assert Le(u, 0.0).on(Omega).rel is Relation.LE

    def test_greater_equal(self):
        assert Ge(u, 0.0).on(GD).rel is Relation.GE

    def test_complementarity(self):
        assert Complementary(u, 0.0).on(Omega).rel is Relation.COMPLEMENTARITY


class TestConstraintValidation:
    def test_sides_must_be_expressions(self):
        with pytest.raises(TypeError):
            Constraint(u, Relation.EQ, 1.0, Omega)

    def test_relation_must_be_a_relation(self):
        with pytest.raises(TypeError):
            Constraint(u, "=", Const(1.0), Omega)

    def test_set_must_be_a_set(self):
        with pytest.raises(TypeError):
            Constraint(u, Relation.EQ, Const(1.0), "Omega")
