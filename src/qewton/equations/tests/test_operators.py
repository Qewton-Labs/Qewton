import pytest

from qewton.config.variables import Variable
from qewton.equations.expr.nodes import Apply, Const, Coord, FunctionSymbol
from qewton.equations.expr.ops import OPS, OpSpec, ShapeError, register

x = Variable("x", dim=2)
t = Variable("t", dim=1)
y3 = Variable("y", dim=3)
u = FunctionSymbol(Variable("u"), x)
v = FunctionSymbol(Variable("v"), x)
nu = FunctionSymbol(Variable("nu"), None)


class TestRegistry:
    def test_known_operators_are_registered(self):
        names = ["add", "sub", "neg", "mul", "dot", "inner", "pow", "exp", "sin", "cos"]
        names += ["d", "grad", "div", "curl", "lap"]
        assert all(name in OPS for name in names)

    def test_names_are_unique(self):
        with pytest.raises(ValueError):
            register(OpSpec("add", 2, OPS["add"].shape_rule))

    def test_orders_count_derivatives(self):
        assert OPS["add"].order == 0
        assert OPS["grad"].order == 1
        assert OPS["lap"].order == 2

    def test_linearity_labels(self):
        assert OPS["grad"].linearity == "linear"
        assert OPS["mul"].linearity == "multilinear"
        assert OPS["sin"].linearity == "nonlinear"


class TestShapeRules:
    def test_elementwise_broadcasts_scalars(self):
        assert OPS["add"].shape_rule(((2,), ()), None) == (2,)
        assert OPS["add"].shape_rule(((), (2,)), None) == (2,)

    def test_elementwise_rejects_different_shapes(self):
        with pytest.raises(ShapeError):
            OPS["add"].shape_rule(((2,), (3,)), None)

    def test_mul_needs_one_scalar(self):
        assert OPS["mul"].shape_rule(((), (2,)), None) == (2,)
        with pytest.raises(ShapeError):
            OPS["mul"].shape_rule(((2,), (2,)), None)

    def test_dot_of_vectors_is_a_scalar(self):
        assert OPS["dot"].shape_rule(((2,), (2,)), None) == ()
        with pytest.raises(ShapeError):
            OPS["dot"].shape_rule(((2,), (3,)), None)

    def test_dot_contracts_matrices_with_vectors_and_matrices(self):
        assert OPS["dot"].shape_rule(((2, 2), (2,)), None) == (2,)
        assert OPS["dot"].shape_rule(((2,), (2, 2)), None) == (2,)
        assert OPS["dot"].shape_rule(((2, 2), (2, 2)), None) == (2, 2)
        with pytest.raises(ShapeError):
            OPS["dot"].shape_rule(((2, 3), (2,)), None)

    def test_dot_needs_non_scalars(self):
        with pytest.raises(ShapeError):
            OPS["dot"].shape_rule(((), (2,)), None)

    def test_inner_contracts_equal_shapes(self):
        assert OPS["inner"].shape_rule(((2, 2), (2, 2)), None) == ()
        with pytest.raises(ShapeError):
            OPS["inner"].shape_rule(((2,), (3,)), None)

    def test_scalar_functions_need_scalars(self):
        assert OPS["sin"].shape_rule(((),), None) == ()
        assert OPS["pow"].shape_rule(((), ()), None) == ()
        with pytest.raises(ShapeError):
            OPS["sin"].shape_rule(((2,),), None)

    def test_partial_needs_a_single_coordinate(self):
        assert OPS["d"].shape_rule(((),), t) == ()
        assert OPS["d"].shape_rule(((2,),), t) == (2,)
        with pytest.raises(ShapeError):
            OPS["d"].shape_rule(((),), x)

    def test_gradient_appends_the_total_dimension(self):
        assert OPS["grad"].shape_rule(((),), x) == (2,)
        assert OPS["grad"].shape_rule(((2,),), x) == (2, 2)
        assert OPS["grad"].shape_rule(((),), x * t) == (3,)

    def test_divergence_contracts_the_last_axis(self):
        assert OPS["div"].shape_rule(((2,),), x) == ()
        assert OPS["div"].shape_rule(((2, 2),), x) == (2,)
        with pytest.raises(ShapeError):
            OPS["div"].shape_rule(((),), x)
        with pytest.raises(ShapeError):
            OPS["div"].shape_rule(((3,),), x)

    def test_curl_of_a_3d_vector(self):
        assert OPS["curl"].shape_rule(((3,),), y3) == (3,)
        with pytest.raises(ShapeError):
            OPS["curl"].shape_rule(((2,),), x)

    def test_laplacian_keeps_the_shape(self):
        assert OPS["lap"].shape_rule(((2,),), x) == (2,)


class TestApply:
    def test_builds_from_expression_arguments(self):
        assert Apply(OPS["add"], (u, Const(1.0))).args[1] == Const(1.0)

    def test_wrong_number_of_arguments_is_rejected(self):
        with pytest.raises(ValueError):
            Apply(OPS["add"], (u,))

    def test_derivative_needs_its_axes(self):
        with pytest.raises(ValueError):
            Apply(OPS["grad"], (u,))

    def test_axes_must_be_coordinates(self):
        with pytest.raises(ValueError):
            Apply(OPS["grad"], (u, Const(1.0)))

    def test_axes_are_not_allowed_for_pointwise_operators(self):
        with pytest.raises(ValueError):
            Apply(OPS["add"], (u, v, x))

    def test_coordinate_is_not_an_expression_argument(self):
        with pytest.raises(TypeError):
            Apply(OPS["add"], (u, x))

    def test_same_arguments_are_equal(self):
        assert Apply(OPS["grad"], (u, x)) == Apply(OPS["grad"], (u, x))

    def test_argument_order_matters(self):
        assert Apply(OPS["add"], (u, v)) != Apply(OPS["add"], (v, u))

    def test_applications_are_hashable(self):
        assert hash(Apply(OPS["grad"], (u, x))) == hash(Apply(OPS["grad"], (u, x)))


class TestOverloads:
    def test_expression_plus_number(self):
        assert u + 1 == Apply(OPS["add"], (u, Const(1)))

    def test_number_plus_expression(self):
        assert 1 + u == Apply(OPS["add"], (Const(1), u))

    def test_subtraction_keeps_order(self):
        assert u - v == Apply(OPS["sub"], (u, v))
        assert 1 - u == Apply(OPS["sub"], (Const(1), u))

    def test_negation(self):
        assert -u == Apply(OPS["neg"], (u,))

    def test_multiplication_by_number(self):
        assert 2 * u == Apply(OPS["mul"], (Const(2), u))
        assert u * 2 == Apply(OPS["mul"], (u, Const(2)))

    def test_division_is_multiplication_by_a_power_of_minus_one(self):
        assert u / 2 == Apply(OPS["mul"], (u, Apply(OPS["pow"], (Const(2), Const(-1)))))
        assert 1 / u == Apply(OPS["mul"], (Const(1), Apply(OPS["pow"], (u, Const(-1)))))

    def test_powers(self):
        assert u**2 == Apply(OPS["pow"], (u, Const(2)))
        assert 2**u == Apply(OPS["pow"], (Const(2), u))

    def test_strings_and_booleans_are_not_expressions(self):
        with pytest.raises(TypeError):
            u + "a"
        with pytest.raises(TypeError):
            u + True

    def test_equal_expressions_are_equal(self):
        assert (u + 1) == (u + 1)

    def test_the_first_term_of_the_worked_example(self):
        """-div(nu * grad(u, x), x) built from the registry."""
        flux = nu * Apply(OPS["grad"], (u, x))
        term = -Apply(OPS["div"], (flux, x))
        assert term == Apply(OPS["neg"], (Apply(OPS["div"], (flux, x)),))


class TestCoordinateOperands:
    def test_one_dimensional_coordinate_on_the_left(self):
        assert t * u == Apply(OPS["mul"], (Coord(t, 0), u))

    def test_one_dimensional_coordinate_on_the_right(self):
        assert u * t == Apply(OPS["mul"], (u, Coord(t, 0)))

    def test_coordinate_plus_expression(self):
        assert t + u == Apply(OPS["add"], (Coord(t, 0), u))

    def test_multi_component_coordinate_needs_a_component(self):
        with pytest.raises(TypeError):
            x * u

    def test_coordinate_products_are_still_variables(self):
        product = x * t
        assert isinstance(product, Variable)
        assert product.dim == 3
        assert x + t == x * t
