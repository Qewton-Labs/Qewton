import pytest

from qewton.equations import (
    Domain,
    Eq,
    FunctionSymbol,
    Problem,
    ShapeError,
    Variable,
    cos,
    curl,
    div,
    dot,
    dt,
    grad,
    inner,
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
from qewton.equations.analysis.infer import infer_shapes

x = Variable("x", dim=2)
t = Variable("t", dim=1)
Omega = Domain("Omega", ambient=x)


def sym(name, inputs=x, dim=None):
    return FunctionSymbol(Variable(name, dim=dim), inputs)


def infer(*constraints):
    return infer_shapes(Problem(constraints=constraints))


def by_name(result):
    return {s.name: shape for s, shape in result.shapes.items()}


class TestPropagation:
    def test_known_right_side_fixes_the_left(self):
        u, f = sym("u"), sym("f", dim=1)
        result = infer(Eq(lap(u, x), f).on(Omega))
        assert by_name(result) == {"u": (), "f": ()}
        assert result.defaulted == ()

    def test_vector_through_gradient(self):
        u, f = sym("u"), sym("f", dim=2)
        assert by_name(infer(Eq(grad(u, x), f).on(Omega)))["u"] == ()

    def test_gradient_of_a_vector_gives_a_matrix(self):
        u, K = sym("u"), sym("K", dim=(3, 2))
        assert by_name(infer(Eq(grad(u, x), K).on(Omega)))["u"] == (3,)

    def test_divergence_adds_the_axis(self):
        q, f = sym("q"), sym("f", dim=1)
        assert by_name(infer(Eq(div(q, x), f).on(Omega)))["q"] == (2,)

    def test_scalar_only_operators_force_scalars(self):
        u = sym("u")
        result = infer(Eq(cos(u), 0).on(Omega))
        assert by_name(result) == {"u": ()}
        assert result.defaulted == ()

    def test_shift_offset_gets_its_shape(self):
        u, tau = sym("u", t), sym("tau", None)
        result = infer(Eq(dt(u, t), shift(u, t, tau)).on(Domain("I", ambient=t)))
        assert by_name(result)["tau"] == ()

    def test_inner_takes_the_other_shape(self):
        a, b, f = sym("a", dim=3), sym("b"), sym("f", dim=1)
        assert by_name(infer(Eq(inner(a, b), f).on(Omega)))["b"] == (3,)

    def test_scalar_times_vector(self):
        nu, v, f = sym("nu"), sym("v", dim=2), sym("f", dim=2)
        assert by_name(infer(Eq(nu * v, f).on(Omega)))["nu"] == ()

    def test_sum_with_a_known_scalar_side(self):
        a, f = sym("a"), sym("f", dim=2)
        assert by_name(infer(Eq(a + 1, f).on(Omega)))["a"] == (2,)


class TestScalarConstants:
    def test_a_scalar_constant_does_not_decide(self):
        v = sym("v")
        result = infer(Eq(v, 0).on(Omega))
        assert result.defaulted == (v,)

    def test_a_scalar_constant_fits_a_known_vector(self):
        v = sym("v", dim=2)
        assert by_name(infer(Eq(v, 0).on(Omega))) == {"v": (2,)}


class TestDefaults:
    def test_defaults_are_reported_and_resolve_the_rest(self):
        nu, u, f = sym("nu", None), sym("u"), sym("f", None)
        result = infer(Eq(-div(nu * grad(u, x), x), f).on(Omega))
        # Both sides are unknown, so nothing decides u after nu: it is defaulted too.
        assert result.defaulted == (nu, u)
        assert by_name(result) == {"nu": (), "u": (), "f": ()}

    def test_no_default_when_everything_follows(self):
        u = sym("u", dim=1)
        assert infer(Eq(lap(u, x), 1).on(Omega)).defaulted == ()


class TestConflicts:
    def test_symbol_used_with_two_shapes(self):
        u = sym("u")
        a, b = sym("a", dim=1), sym("b", dim=1)
        with pytest.raises(ShapeError):
            infer(Eq(lap(u, x), a).on(Omega), Eq(div(u, x), b).on(Omega))

    def test_wrong_gradient_length(self):
        u, f = sym("u"), sym("f", dim=(3,))
        with pytest.raises(ShapeError):
            infer(Eq(grad(u, x), f).on(Omega))

    def test_operator_mismatch_after_inference(self):
        q, f = sym("q"), sym("f", dim=1)
        with pytest.raises(ShapeError):
            infer(Eq(curl(q, x), f).on(Omega))

    def test_dot_does_not_guess(self):
        # dot is ambiguous, so the unknown is defaulted rather than inferred
        a, b = sym("a", dim=(2, 2)), sym("b")
        f = sym("f", dim=2)
        with pytest.raises(ShapeError):
            infer(Eq(dot(a, b), f).on(Omega))


class TestFixtures:
    @pytest.mark.parametrize(
        "make", [poisson, heat, neumann_poisson, delay, periodic_poisson]
    )
    def test_every_symbol_gets_a_shape(self, make):
        problem = make()
        result = infer_shapes(problem)
        assert set(result.shapes) == problem.symbols()
        assert all(shape == () for shape in result.shapes.values())

    def test_poisson_declares_everything_that_is_ambiguous(self):
        assert infer_shapes(poisson()).defaulted == ()

    def test_mean_of_a_vector_symbol(self):
        p = sym("p", dim=2)
        assert by_name(infer(Eq(mean(p, Omega), (0, 0)).on(None)))["p"] == (2,)
