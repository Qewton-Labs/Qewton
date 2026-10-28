import pytest

from qewton.config.variables import Variable
from qewton.equations.expr.nodes import Const, Coord, FunctionSymbol, Normal, Restrict
from qewton.equations.geometry.sets import Boundary, Domain, Part, Product, Slice

x = Variable("x", dim=2)
t = Variable("t", dim=1)
Omega = Domain("Omega", ambient=x)
dO = Boundary(Omega)
Gamma_D = Part("Gamma_D", dO)


class TestFunctionSymbol:
    def test_symbol_equals_itself(self):
        u = FunctionSymbol(Variable("u"), x)
        assert u == u

    def test_symbols_with_the_same_variable_are_different(self):
        assert FunctionSymbol(Variable("u"), x) != FunctionSymbol(Variable("u"), x)

    def test_symbols_are_hashable_by_identity(self):
        u = FunctionSymbol(Variable("u"), x)
        assert len({u, u, FunctionSymbol(Variable("u"), x)}) == 2

    def test_name_comes_from_the_variable(self):
        assert FunctionSymbol(Variable("u"), x).name == "u"

    def test_output_must_be_a_variable(self):
        with pytest.raises(TypeError):
            FunctionSymbol("u", x)


class TestOutputShape:
    def test_unknown_shape_is_none(self):
        assert FunctionSymbol(Variable("u"), x).shape is None

    def test_dim_one_is_a_scalar(self):
        assert FunctionSymbol(Variable("f", dim=1), None).shape == ()

    def test_dim_n_is_a_vector(self):
        assert FunctionSymbol(Variable("u", dim=2), x).shape == (2,)

    def test_tuple_dim_is_a_tensor(self):
        assert FunctionSymbol(Variable("T", dim=(3, 3)), x).shape == (3, 3)

    def test_output_dims_must_be_positive(self):
        with pytest.raises(ValueError):
            FunctionSymbol(Variable("u", dim=0), x)


class TestInputs:
    def test_inputs_must_always_be_given(self):
        with pytest.raises(TypeError):
            FunctionSymbol(Variable("u"))

    def test_none_declares_a_constant(self):
        nu = FunctionSymbol(Variable("nu"), None)
        assert nu.inputs is None

    def test_product_of_coordinates_declares_inputs(self):
        u = FunctionSymbol(Variable("u", dim=2), x * t)
        assert u.inputs == x * t

    def test_order_of_the_product_is_the_input_layout(self):
        xt = FunctionSymbol(Variable("u"), x * t).inputs
        tx = FunctionSymbol(Variable("u"), t * x).inputs
        assert xt != tx
        assert [leaf.name for leaf in xt.leaves] == ["x_1", "x_2", "t"]
        assert [leaf.name for leaf in tx.leaves] == ["t", "x_1", "x_2"]

    def test_single_coordinate_is_accepted(self):
        assert FunctionSymbol(Variable("u"), x).inputs == x

    def test_zero_dimensional_variable_is_not_a_constant(self):
        with pytest.raises(ValueError):
            FunctionSymbol(Variable("nu"), Variable(dim=0))

    def test_inputs_must_be_coordinates(self):
        with pytest.raises(ValueError):
            FunctionSymbol(Variable("u"), Variable("q"))

    def test_inputs_must_be_a_variable_not_a_set(self):
        with pytest.raises(TypeError):
            FunctionSymbol(Variable("u"), {x})


class TestCoord:
    def test_same_component_is_equal(self):
        assert Coord(x, 0) == Coord(x, 0)
        assert hash(Coord(x, 0)) == hash(Coord(x, 0))

    def test_different_components_differ(self):
        assert Coord(x, 0) != Coord(x, 1)

    def test_component_must_exist(self):
        with pytest.raises(ValueError):
            Coord(x, 2)
        with pytest.raises(ValueError):
            Coord(x, -1)

    def test_one_dimensional_coordinate_has_one_component(self):
        assert Coord(t, 0).component == 0
        with pytest.raises(ValueError):
            Coord(t, 1)

    def test_needs_a_coordinate(self):
        with pytest.raises(ValueError):
            Coord(Variable("q"), 0)


class TestConst:
    def test_equal_values_are_equal(self):
        assert Const(1.0) == Const(1)
        assert hash(Const(1.0)) == hash(Const(1))

    def test_rejects_non_numbers(self):
        with pytest.raises(TypeError):
            Const("1")
        with pytest.raises(TypeError):
            Const(True)

    def test_vector_and_matrix_shapes(self):
        assert Const((0.0, -9.81)).shape == (2,)
        assert Const([[1.0, 0.0], [0.0, 2.0]]).shape == (2, 2)
        assert Const(1.0).shape == ()

    def test_lists_and_tuples_are_the_same_constant(self):
        assert Const([1.0, 2.0]) == Const((1.0, 2.0))
        assert hash(Const([1.0, 2.0])) == hash(Const((1.0, 2.0)))

    def test_ragged_matrices_are_rejected(self):
        with pytest.raises(ValueError):
            Const([[1.0, 0.0], [0.0]])

    def test_empty_vectors_are_rejected(self):
        with pytest.raises(ValueError):
            Const([])

    def test_vector_entries_must_be_numbers(self):
        with pytest.raises(TypeError):
            Const([1.0, "2"])
        with pytest.raises(TypeError):
            Const([True, 1.0])


class TestNormal:
    def test_normal_on_a_boundary(self):
        assert Normal(dO).of == dO

    def test_normal_on_a_part_of_the_boundary(self):
        assert Normal(Gamma_D).of == Gamma_D

    def test_no_normal_in_the_interior(self):
        with pytest.raises(ValueError):
            Normal(Omega)

    def test_no_normal_on_a_slice(self):
        with pytest.raises(ValueError):
            Normal(Slice(t, 0.0))


class TestRestrict:
    def test_same_expression_and_set_are_equal(self):
        u = FunctionSymbol(Variable("u"), x)
        assert Restrict(u, Gamma_D) == Restrict(u, Gamma_D)

    def test_different_sets_differ(self):
        u = FunctionSymbol(Variable("u"), x)
        assert Restrict(u, Gamma_D) != Restrict(u, Omega)

    def test_restriction_to_a_slice_of_space_time(self):
        u = FunctionSymbol(Variable("u"), x * t)
        initial = Restrict(u, Product((Omega, Slice(t, 0.0))))
        assert hash(initial) == hash(Restrict(u, Product((Omega, Slice(t, 0.0)))))

    def test_needs_an_expression_and_a_set(self):
        u = FunctionSymbol(Variable("u"), x)
        with pytest.raises(TypeError):
            Restrict("u", Omega)
        with pytest.raises(TypeError):
            Restrict(u, "Omega")
