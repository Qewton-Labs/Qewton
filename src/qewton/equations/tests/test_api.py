import pytest

from qewton.config.variables import Variable
from qewton.equations.expr import api
from qewton.equations.expr.nodes import (
    Apply,
    Coord,
    Const,
    FunctionSymbol,
    Integral,
    Shift,
    children,
)
from qewton.equations.expr.ops import OPS
from qewton.equations.geometry.sets import Domain
from qewton.equations.analysis.shapes import shape_of

x = Variable("x", dim=2)
t = Variable("t", dim=1)
y3 = Variable("y", dim=3)
u = FunctionSymbol(Variable("u"), x)
v = FunctionSymbol(Variable("v"), x)
u_xt = FunctionSymbol(Variable("u"), x * t)
w = FunctionSymbol(Variable("w", dim=3), y3)
nu = FunctionSymbol(Variable("nu"), None)


class TestDerivatives:
    def test_gradient_takes_the_axes(self):
        assert api.grad(u, x) == Apply(OPS["grad"], (u, x))

    def test_gradient_over_space_and_time(self):
        assert api.grad(u_xt, x * t) == Apply(OPS["grad"], (u_xt, x * t))

    def test_divergence(self):
        assert api.div(v, x) == Apply(OPS["div"], (v, x))

    def test_curl(self):
        assert api.curl(w, y3) == Apply(OPS["curl"], (w, y3))

    def test_laplacian(self):
        assert api.lap(u, x) == Apply(OPS["lap"], (u, x))

    def test_partial_derivative(self):
        assert api.d(u_xt, t) == Apply(OPS["d"], (u_xt, t))

    def test_time_derivative_is_the_partial_derivative_along_time(self):
        assert api.dt(u_xt, t) == api.d(u_xt, t)

    def test_derivatives_need_their_axes(self):
        with pytest.raises(TypeError):
            api.grad(u)

    def test_axes_must_be_coordinates(self):
        with pytest.raises(ValueError):
            api.div(v, Const(1.0))


class TestFunctions:
    def test_scalar_functions(self):
        assert api.sin(u) == Apply(OPS["sin"], (u,))
        assert api.cos(u) == Apply(OPS["cos"], (u,))
        assert api.exp(u) == Apply(OPS["exp"], (u,))

    def test_numbers_become_constants(self):
        assert api.sin(2.0) == Apply(OPS["sin"], (Const(2.0),))

    def test_one_dimensional_coordinate_becomes_its_component(self):
        assert api.sin(t) == Apply(OPS["sin"], (Coord(t, 0),))

    def test_multi_component_coordinate_is_rejected(self):
        with pytest.raises(TypeError):
            api.sin(x)

    def test_products_of_expressions(self):
        assert api.dot(u, v) == Apply(OPS["dot"], (u, v))
        assert api.inner(u, v) == Apply(OPS["inner"], (u, v))

    def test_products_need_two_arguments(self):
        with pytest.raises(TypeError):
            api.dot(u)


class TestConstantVectorsAndMatrices:
    def test_matrix_times_gradient_is_a_matrix_vector_product(self):
        K = [[1.0, 0.0], [0.0, 2.0]]
        assert api.dot(K, api.grad(u, x)) == Apply(
            OPS["dot"],
            (Const(((1.0, 0.0), (0.0, 2.0))), Apply(OPS["grad"], (u, x))),
        )

    def test_vector_constant_in_an_expression(self):
        assert u * [0.0, -9.81] == Apply(OPS["mul"], (u, Const((0.0, -9.81))))


class TestWorkedExample:
    def test_divergence_of_the_flux(self):
        """div(nu * grad(u, x), x), the left-hand side of the worked example."""
        flux = nu * api.grad(u, x)
        assert api.div(flux, x) == Apply(
            OPS["div"], (Apply(OPS["mul"], (nu, Apply(OPS["grad"], (u, x)))), x)
        )


class TestIntegrals:
    omega = Domain("Omega", ambient=x)
    p = FunctionSymbol(Variable("p", dim=1), x)
    q = FunctionSymbol(Variable("q", dim=2), x)

    def test_integral_builds_and_prints(self):
        assert str(api.integral(self.p, self.omega)) == "∫_{Omega} p"
        assert str(api.integral(self.p * self.p, self.omega)) == "∫_{Omega} (p·p)"

    def test_mean_is_integral_over_measure(self):
        assert str(api.mean(self.p, self.omega)) == "∫_{Omega} p / ∫_{Omega} 1"

    def test_integral_keeps_the_shape(self):
        assert shape_of(api.integral(self.q, self.omega)) == (2,)

    def test_integral_checks_its_types(self):
        with pytest.raises(TypeError):
            Integral(self.p, "Omega")
        with pytest.raises(TypeError):
            Integral(1.0, self.omega)

    def test_children_is_the_integrand(self):
        assert children(api.integral(self.q, self.omega)) == (self.q,)


class TestShift:
    u1 = FunctionSymbol(Variable("u", dim=1), x * t)
    tau = FunctionSymbol(Variable("tau", dim=1), None)

    def test_delay_prints_like_the_math(self):
        assert str(api.shift(self.u1, t, -1)) == "u(t − 1)"
        assert str(api.shift(self.u1, t, 2)) == "u(t + 2)"
        assert str(api.shift(self.u1, t, self.tau)) == "u(t + tau)"

    def test_vector_offset_for_periodicity(self):
        assert str(api.shift(self.u1, x, (-1, 0))) == "u(x + (−1, 0))"

    def test_delay_equation_builds(self):
        from qewton.equations.model.constraints import Eq

        eq = Eq(api.dt(self.u1, t), api.shift(self.u1, t, -1))
        assert str(eq.lhs) == "∂u/∂t"

    def test_offset_must_be_constant(self):
        with pytest.raises(ValueError, match="constant"):
            api.shift(self.u1, t, Coord(t, 0))
        with pytest.raises(ValueError, match="constant"):
            api.shift(self.u1, t, FunctionSymbol(Variable("s", dim=1), t))

    def test_offset_shape_follows_the_axis(self):
        from qewton.equations.expr.ops import ShapeError

        with pytest.raises(ShapeError):
            api.shift(self.u1, x, 1)
        with pytest.raises(ShapeError):
            api.shift(self.u1, t, (1, 2))
        with pytest.raises(ShapeError):
            api.shift(self.u1, x, (1, 2, 3))

    def test_keeps_shape_and_children(self):
        shifted = api.shift(self.u1, t, self.tau)
        assert shape_of(shifted) == ()
        assert children(shifted) == (self.u1, self.tau)
        assert isinstance(shifted, Shift)

    def test_axis_must_be_a_coordinate(self):
        with pytest.raises(Exception):
            api.shift(self.u1, Variable("u"), 1)
