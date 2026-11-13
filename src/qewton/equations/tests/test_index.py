import pytest

from qewton.equations import (
    Domain,
    Eq,
    FunctionSymbol,
    Index,
    Problem,
    Variable,
    div,
    grad,
    lap,
    transform,
    walk,
)
from qewton.equations.analysis.checks.axes import free_axes
from qewton.equations.analysis.checks.shapes import UnknownShapeError, shape_of
from qewton.equations.analysis.prepare.infer import infer_shapes
from qewton.equations.analysis.context import signatures
from qewton.equations.analysis.structure.structure import linearity
from qewton.equations.expr.ops import ShapeError
from qewton.equations.printing.pretty import pretty

x = Variable("x", dim=2)
t = Variable("t", dim=1)
u = FunctionSymbol(Variable("u", dim=1), x)
v = FunctionSymbol(Variable("v", dim=2), x)
K = FunctionSymbol(Variable("K", dim=(2, 3)), x)
free = FunctionSymbol(Variable("w"), x)


class TestBuilding:
    def test_brackets_build_an_index(self):
        assert v[0] == Index(v, 0)

    def test_tuple_indexes_one_axis_after_another(self):
        assert K[1, 2] == Index(Index(K, 1), 2)
        assert K[1, 2] == K[1][2]

    def test_children_and_rebuild(self):
        node = v[1]
        assert node.children() == (v,)
        assert node.with_children((v + v,)) == Index(v + v, 1)

    def test_transform_reaches_the_operand(self):
        result = transform(v[0], lambda node: v if node == v else node)
        assert result == v[0]
        assert list(walk(v[0])) == [v[0], v]

    def test_expression_is_not_iterable(self):
        with pytest.raises(TypeError):
            tuple(v)

    def test_index_must_be_a_non_negative_integer(self):
        with pytest.raises(ValueError, match="from 0"):
            v[-1]
        with pytest.raises(TypeError, match="integer"):
            v[0.5]
        with pytest.raises(TypeError, match="integer"):
            v[True]


class TestShape:
    def test_vector_component_is_a_scalar(self):
        assert shape_of(v[0]) == ()

    def test_matrix_row_and_entry(self):
        assert shape_of(K[0]) == (3,)
        assert shape_of(K[0, 2]) == ()

    def test_component_of_gradient(self):
        assert shape_of(grad(u, x)[1]) == ()

    def test_scalar_has_no_components(self):
        with pytest.raises(ShapeError, match="scalar"):
            u[0]

    def test_out_of_range_is_caught_when_built(self):
        with pytest.raises(ShapeError, match="out of range"):
            v[2]
        with pytest.raises(ShapeError, match="out of range"):
            K[0, 3]

    def test_component_sits_in_scalar_operators(self):
        assert shape_of(div(v, x) + v[0]) == ()

    def test_undeclared_shape_is_not_checked_yet(self):
        assert free[5] == Index(free, 5)
        with pytest.raises(UnknownShapeError):
            shape_of(free[5])


class TestAxes:
    def test_axes_of_the_operand(self):
        assert free_axes(v[0]) == frozenset({x})


class TestPrinting:
    def test_single_index(self):
        assert pretty(v[0]) == "v[0]"

    def test_chain_is_merged(self):
        assert pretty(K[1, 2]) == "K[1, 2]"

    def test_index_binds_tighter_than_an_operator(self):
        assert pretty(v[0] + 2 * v[1]) == "v[0] + 2·v[1]"

    def test_index_of_a_compound(self):
        assert pretty((v + v)[0]) == "(v + v)[0]"

    def test_in_a_constraint(self):
        assert pretty(Eq(v[0], 0)) == "v[0] = 0  (no set yet)"


Omega = Domain("Omega", ambient=x)
q = FunctionSymbol(Variable("q"), x)
f = FunctionSymbol(Variable("f", dim=1), x)
g = FunctionSymbol(Variable("g", dim=2), x)


class TestInference:
    def test_other_constraints_fix_the_shape(self):
        problem = Problem(constraints=[Eq(q[1], f).on(Omega), Eq(q, g).on(Omega)])
        assert infer_shapes(problem).shapes[q] == (2,)

    def test_component_out_of_range_after_inference(self):
        problem = Problem(constraints=[Eq(q[2], f).on(Omega), Eq(q, g).on(Omega)])
        with pytest.raises(ShapeError, match="out of range"):
            infer_shapes(problem)

    def test_a_component_alone_cannot_fix_the_shape(self):
        problem = Problem(constraints=[Eq(q[0], f).on(Omega)])
        with pytest.raises(ShapeError, match="declare it"):
            infer_shapes(problem)


class TestAnalyses:
    def test_symbols_inside_a_component_are_found(self):
        problem = Problem(constraints=[Eq(v[0], f).on(Omega)])
        assert set(signatures(problem.task()).table) == {v, f}

    def test_linearity_passes_through(self):
        assert linearity(v[0], frozenset({v})) == "linear"
        assert linearity(v[0] * v[1], frozenset({v})) == "nonlinear"
        assert linearity(v[0], frozenset()) == "none"
