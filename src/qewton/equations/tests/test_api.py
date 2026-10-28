import pytest

from qewton.config.variables import Variable
from qewton.equations.expr import api
from qewton.equations.expr.nodes import Apply, Coord, Const, FunctionSymbol
from qewton.equations.expr.ops import OPS

x = Variable("x", dim=2)
t = Variable("t", dim=1)
y3 = Variable("y", dim=3)
u = FunctionSymbol(Variable("u"), x)
v = FunctionSymbol(Variable("v"), x)
w = FunctionSymbol(Variable("w", dim=3), y3)
nu = FunctionSymbol(Variable("nu"), None)


class TestDerivatives:
    def test_gradient_takes_the_axes(self):
        assert api.grad(u, x) == Apply(OPS["grad"], (u, x))

    def test_gradient_over_space_and_time(self):
        assert api.grad(u, x * t) == Apply(OPS["grad"], (u, x * t))

    def test_divergence(self):
        assert api.div(v, x) == Apply(OPS["div"], (v, x))

    def test_curl(self):
        assert api.curl(w, y3) == Apply(OPS["curl"], (w, y3))

    def test_laplacian(self):
        assert api.lap(u, x) == Apply(OPS["lap"], (u, x))

    def test_partial_derivative(self):
        assert api.d(u, t) == Apply(OPS["d"], (u, t))

    def test_time_derivative_is_the_partial_derivative_along_time(self):
        assert api.dt(u, t) == api.d(u, t)

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
