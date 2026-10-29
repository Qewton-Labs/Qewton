import pytest

from qewton.config.variables import Variable
from qewton.equations.model.constraints import Eq
from qewton.equations.expr import api
from qewton.equations.expr.nodes import Const, Coord, FunctionSymbol, Normal, Restrict
from qewton.equations.geometry.sets import (
    Boundary,
    Domain,
    Part,
    Partition,
    Product,
    Slice,
)
from qewton.equations.model.problem import Problem
from qewton.equations.printing.pretty import pretty

x = Variable("x", dim=2)
t = Variable("t", dim=1)
Omega = Domain("Omega", ambient=x)
Interval = Domain("I", ambient=t)
dO = Boundary(Omega)
GD = Part("Gamma_D", dO)
GN = Part("Gamma_N", dO)
u = FunctionSymbol(Variable("u"), x)
u_xt = FunctionSymbol(Variable("u"), x * t)
v = FunctionSymbol(Variable("v"), x)
f = FunctionSymbol(Variable("f"), None)
nu = FunctionSymbol(Variable("nu"), None)


class TestArithmetic:
    def test_symbol(self):
        assert pretty(u) == "u"

    def test_sum_and_difference(self):
        assert pretty(u + 1) == "u + 1"
        assert pretty(u - 1) == "u − 1"
        assert pretty(1 - u) == "1 − u"

    def test_scalar_product_has_a_dot(self):
        assert pretty(2 * u) == "2·u"

    def test_negation(self):
        assert pretty(-u) == "−u"
        assert pretty(-(u + 1)) == "−(u + 1)"

    def test_powers_use_superscripts(self):
        assert pretty(u**2) == "u²"
        assert pretty((u + 1) ** 2) == "(u + 1)²"
        assert pretty(2**u) == "2^u"

    def test_division_is_printed_as_a_fraction(self):
        assert pretty(u / 2) == "u / 2"
        assert pretty(1 / u) == "1 / u"

    def test_sums_inside_products_are_parenthesized(self):
        assert pretty((u + 1) * v) == "(u + 1)·v"

    def test_negative_constant_as_a_factor(self):
        assert pretty(-1.5 * u) == "−1.5·u"


class TestCoordinatesAndConstants:
    def test_one_dimensional_coordinate(self):
        assert pretty(Coord(t, 0)) == "t"

    def test_components_are_subscripted_from_one(self):
        assert pretty(Coord(x, 0)) == "x₁"
        assert pretty(Coord(x, 1)) == "x₂"

    def test_vector_and_matrix_constants(self):
        assert pretty(Const((0.0, -9.81))) == "(0, −9.81)"
        assert pretty(Const(((1.0, 0.0), (0.0, 2.0)))) == "[(1, 0), (0, 2)]"


class TestOperators:
    def test_gradient(self):
        assert pretty(api.grad(u, x)) == "∇_x u"

    def test_divergence_of_a_product_is_parenthesized(self):
        flux = nu * api.grad(u, x)
        assert pretty(api.div(flux, x)) == "∇_x·(nu·∇_x u)"
        assert pretty(-api.div(flux, x)) == "−∇_x·(nu·∇_x u)"

    def test_partial_derivative(self):
        assert pretty(api.d(u_xt, t)) == "∂u/∂t"
        assert pretty(api.dt(u_xt, t) - api.lap(u_xt, x)) == "∂u/∂t − Δ_x u"

    def test_functions_are_called_by_name(self):
        assert pretty(api.sin(u)) == "sin(u)"
        assert pretty(api.dot([[1.0, 0.0], [0.0, 2.0]], api.grad(u, x))) == (
            "dot([(1, 0), (0, 2)], ∇_x u)"
        )


class TestRestrictionsAndNormals:
    def test_restriction_to_a_part(self):
        assert pretty(Restrict(u, GD)) == "u|_{Gamma_D}"

    def test_normal_on_the_boundary(self):
        assert pretty(Normal(dO)) == "n_{∂Omega}"


class TestSets:
    def test_domain_and_part(self):
        assert pretty(Omega) == "Omega"
        assert pretty(GD) == "Gamma_D"

    def test_boundary(self):
        assert pretty(dO) == "∂Omega"

    def test_product(self):
        assert pretty(Product((Omega, Interval))) == "Omega × I"

    def test_slice_inside_a_product(self):
        assert pretty(Product((Omega, Slice(t, 0.0)))) == "Omega × {t = 0}"

    def test_partition(self):
        assert pretty(Partition(dO, (GD, GN))) == "∂Omega = Gamma_D ⊔ Gamma_N"


class TestConstraintsAndProblems:
    def test_constraint(self):
        constraint = Eq(-api.div(nu * api.grad(u, x), x), f).on(Omega)
        assert pretty(constraint) == "−∇_x·(nu·∇_x u) = f on Omega"

    def test_named_constraint(self):
        assert pretty(Eq(u, 0.0).on(Omega, "pde")) == "pde: u = 0 on Omega"

    def test_pending_constraint_says_it_has_no_set(self):
        assert pretty(Eq(u, 0.0)) == "u = 0  (no set yet)"

    def test_problem_lists_constraints_partitions_and_time(self):
        problem = Problem(
            constraints=(
                Eq(api.dt(u_xt, t) - api.lap(u_xt, x), f).on(Product((Omega, Interval))),
            ),
            partitions=(Partition(dO, (GD, GN)),),
            time=t,
        )
        text = pretty(problem)
        assert text.splitlines() == [
            "constraints:",
            "  1. ∂u/∂t − Δ_x u = f on Omega × I",
            "partitions:",
            "  ∂Omega = Gamma_D ⊔ Gamma_N",
            "time: t",
        ]

    def test_str_and_print_use_the_printer(self):
        assert str(u + 1) == "u + 1"
        assert str(Omega) == "Omega"

    def test_variables_print_their_name(self):
        assert pretty(x) == "x"

    def test_unsupported_objects_are_rejected(self):
        with pytest.raises(TypeError):
            pretty(3)
