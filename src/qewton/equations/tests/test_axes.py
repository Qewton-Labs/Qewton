import pytest

from qewton.equations import (
    AxisError,
    Boundary,
    Domain,
    Eq,
    FunctionSymbol,
    Integral,
    Normal,
    Part,
    Problem,
    Product,
    Restrict,
    Slice,
    Variable,
    dot,
    dt,
    free_axes,
    grad,
    integral,
    lap,
    mean,
    shift,
)
from qewton.equations.fixtures import (
    delay,
    heat,
    neumann_poisson,
    periodic_poisson,
    poisson,
)
from qewton.equations.geometry.coordinates import coordinate_factors, root_of

x = Variable("x", dim=2)
t = Variable("t", dim=1)
y = Variable("y", dim=3)
Omega = Domain("Omega", ambient=x)
Interval = Domain("I", ambient=t)
dO = Boundary(Omega)
u = FunctionSymbol(Variable("u", dim=1), x * t)
f = FunctionSymbol(Variable("f", dim=1), None)


class TestCoordinateFactors:
    def test_single_coordinate(self):
        assert coordinate_factors(x) == (x,)
        assert coordinate_factors(t) == (t,)

    def test_products_are_flattened_in_order(self):
        assert coordinate_factors(x * t) == (x, t)
        assert coordinate_factors(x * t * y) == (x, t, y)
        assert coordinate_factors(x * (t * y)) == (x, t, y)

    def test_a_component_stands_for_its_coordinate(self):
        assert coordinate_factors(x.children[0]) == (x,)
        assert root_of(x.children[1]) == x


class TestFreeAxes:
    def test_functions_follow_their_declared_inputs(self):
        assert free_axes(u) == {x, t}
        assert free_axes(f) == frozenset()

    def test_derivatives_and_arithmetic_keep_the_axes(self):
        assert free_axes(dt(u, t) - lap(u, x) + f) == {x, t}

    def test_coordinates_depend_on_themselves(self):
        assert free_axes(u * Coord_(t)) == {x, t}
        assert free_axes(Coord_(x)) == {x}

    def test_restricting_to_a_time_slice_removes_time(self):
        initial = Product((Omega, Slice(t, 0.0)))
        assert free_axes(Restrict(u, initial)) == {x}

    def test_restricting_to_a_boundary_keeps_space(self):
        assert free_axes(Restrict(u, dO)) == {x, t}

    def test_integrals_remove_the_axes_of_the_set(self):
        assert free_axes(integral(u, Omega)) == {t}
        assert free_axes(mean(u, Omega)) == {t}
        assert free_axes(integral(u, Product((Omega, Interval)))) == frozenset()

    def test_normal_lives_on_the_boundary_coordinates(self):
        assert free_axes(Normal(dO)) == {x}

    def test_shift_keeps_the_axes(self):
        assert free_axes(shift(u, t, -1)) == {x, t}


def Coord_(var):
    from qewton.equations import Coord

    return Coord(var, 0)


class TestEagerChecks:
    def test_derivative_along_an_axis_the_operand_lacks(self):
        spatial = FunctionSymbol(Variable("w", dim=1), x)
        with pytest.raises(AxisError, match="derivative is zero"):
            dt(spatial, t)
        with pytest.raises(AxisError):
            grad(f, x)

    def test_gradient_along_a_product_needs_every_factor(self):
        spatial = FunctionSymbol(Variable("w", dim=1), x)
        with pytest.raises(AxisError, match="{t}"):
            grad(spatial, x * t)

    def test_shift_along_an_axis_the_operand_lacks(self):
        spatial = FunctionSymbol(Variable("w", dim=1), x)
        with pytest.raises(AxisError, match="shift"):
            shift(spatial, t, 1)

    def test_a_derivative_after_an_integral_is_caught(self):
        with pytest.raises(AxisError):
            grad(integral(u, Omega), x)

    def test_a_derivative_of_a_restriction_to_a_slice_is_caught(self):
        initial = Product((Omega, Slice(t, 0.0)))
        with pytest.raises(AxisError):
            dt(Restrict(u, initial), t)


class TestConstraintAxes:
    def test_sides_may_use_fewer_axes_than_the_set(self):
        Problem((Eq(f, 1).on(Omega),))

    def test_a_side_may_not_use_axes_outside_the_set(self):
        w = FunctionSymbol(Variable("w", dim=1), x)
        with pytest.raises(AxisError, match="depends on {x}"):
            Problem((Eq(w, 0).on(Interval),))

    def test_time_is_lifted_when_declared(self):
        # u(x, t) on a purely spatial set is allowed if t is the time of the problem
        Problem((Eq(u, 0).on(Omega),), time=t)
        with pytest.raises(AxisError, match="depends on {t}"):
            Problem((Eq(u, 0).on(Omega),))

    def test_a_global_constraint_has_no_free_axes(self):
        Problem((Eq(mean(u, Product((Omega, Interval))), 0).on(None),))
        with pytest.raises(AxisError, match="globally"):
            Problem((Eq(mean(u, Omega), 0).on(None),))
        with pytest.raises(AxisError, match="Integrate or restrict"):
            Problem((Eq(u, 0).on(None),))

    def test_initial_condition_on_a_slice(self):
        initial = Product((Omega, Slice(t, 0.0)))
        w0 = FunctionSymbol(Variable("w0", dim=1), x)
        Problem((Eq(Restrict(u, initial), w0).on(initial),), time=t)
        with pytest.raises(AxisError):
            Problem((Eq(u, w0).on(initial),))

    def test_integral_node_is_exported_for_construction(self):
        assert isinstance(integral(u, Omega), Integral)


class TestCoordinates:
    def test_poisson_uses_x_only(self):
        assert poisson().coordinates() == {Variable("x", dim=2)}

    def test_heat_uses_space_and_time(self):
        assert heat().coordinates() == {Variable("x", dim=2), Variable("t", dim=1)}

    def test_includes_derivative_and_shift_axes_and_time(self):
        assert delay().coordinates() == {Variable("t", dim=1)}
        assert periodic_poisson().coordinates() == {Variable("x", dim=2)}

    def test_global_problem(self):
        assert neumann_poisson().coordinates() == {Variable("x", dim=2)}

    def test_dot_with_normal(self):
        problem = Problem(
            (Eq(dot(grad(Restrict(u, dO), x), Normal(dO)), 0).on(dO),), time=t
        )
        assert problem.coordinates() == {x, t}


@pytest.mark.parametrize(
    "make", [poisson, heat, neumann_poisson, delay, periodic_poisson]
)
def test_fixtures_pass_the_axis_checks(make):
    make()
