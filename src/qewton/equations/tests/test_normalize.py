from qewton.equations import (
    Boundary,
    Complementary,
    Const,
    Le,
    Relation,
    div,
    grad,
    validate,
    Domain,
    Eq,
    FunctionSymbol,
    Problem,
    Product,
    Restrict,
    Severity,
    Slice,
    Variable,
    dt,
    integral,
    lap,
)
from qewton.equations.analysis.normalize import normalize
from qewton.equations.analysis.diagnostics import errors
from qewton.equations.fixtures import (
    delay,
    heat,
    neumann_poisson,
    periodic_poisson,
    poisson,
)

x = Variable("x", dim=2)
t = Variable("t", dim=1)
Omega = Domain("Omega", ambient=x)
I = Domain("I", ambient=t)
dO = Boundary(Omega)
w = FunctionSymbol(Variable("w", dim=1), x * t)
w0 = FunctionSymbol(Variable("w0", dim=1), x)
g = FunctionSymbol(Variable("g", dim=1), x)
space_time = Product((Omega, I))
initial = Product((Omega, Slice(t, 0.0)))


def heat_with_dirichlet(on=dO, rhs=g, interval=True):
    pde_set = space_time if interval else Omega
    return Problem(
        (
            Eq(dt(w, t) - lap(w, x), 0).on(pde_set),
            Eq(Restrict(w, on), rhs).on(on),
            Eq(Restrict(w, initial), w0).on(initial),
        ),
        time=t,
    )


def ons(problem):
    return [c.on for c in problem.constraints]


def test_a_condition_without_time_is_lifted():
    result = normalize(heat_with_dirichlet())
    lifted = result.problem.constraints[1]
    assert lifted.on == Product((dO, I))
    assert str(lifted.on) == "∂Omega × I"


def test_the_pde_and_the_initial_condition_are_left_alone():
    original = heat_with_dirichlet()
    ons_after = ons(normalize(original).problem)
    assert ons_after[0] == original.constraints[0].on
    assert ons_after[2] == original.constraints[2].on


def test_the_original_is_kept_for_display():
    original = heat_with_dirichlet()
    result = normalize(original)
    assert result.origin == original.constraints
    assert result.origin[1].on == dO


def test_each_lift_is_reported_as_info():
    result = normalize(heat_with_dirichlet())
    assert [d.severity for d in result.diagnostics] == [Severity.INFO]
    assert "Lifted" in result.diagnostics[0].message


def test_time_independent_data_is_not_a_reason_to_lift():
    # Both sides are functions of x only: the condition holds as written.
    c = Problem(
        (
            Eq(dt(w, t), 0).on(space_time),
            Eq(g, 1).on(dO),
        ),
        time=t,
    )
    assert ons(normalize(c).problem) == ons(c)


def test_a_global_constraint_in_time_moves_to_the_interval():
    q = FunctionSymbol(Variable("q", dim=1), x * t)
    problem = Problem(
        (Eq(dt(q, t), 0).on(space_time), Eq(integral(q, Omega), 0).on(None)), time=t
    )
    result = normalize(problem)
    assert result.problem.constraints[1].on == I


def test_a_global_constraint_without_time_stays_global():
    p = FunctionSymbol(Variable("p", dim=1), x)
    problem = Problem((Eq(integral(p, Omega), 0).on(None),), time=t)
    assert ons(normalize(problem).problem) == [None]


def test_without_an_interval_the_lift_is_refused_with_a_warning():
    result = normalize(heat_with_dirichlet(interval=False))
    assert ons(result.problem) == ons(heat_with_dirichlet(interval=False))
    assert [d.severity for d in result.diagnostics] == [Severity.WARNING] * 2
    assert "interval" in result.diagnostics[0].message


def test_two_intervals_are_ambiguous():
    other = Domain("J", ambient=t)
    problem = Problem(
        (
            Eq(dt(w, t), 0).on(space_time),
            Eq(w, 0).on(Product((Omega, other))),
            Eq(Restrict(w, dO), g).on(dO),
        ),
        time=t,
    )
    result = normalize(problem)
    assert ons(result.problem) == ons(problem)
    assert "several intervals" in result.diagnostics[0].message


def test_problems_without_time_are_unchanged():
    result = normalize(poisson())
    assert result.diagnostics == ()
    assert ons(result.problem) == ons(poisson())


def test_heat_needs_no_lifting():
    h = heat()
    assert ons(normalize(h).problem) == ons(h) and normalize(h).diagnostics == ()


def test_the_lifted_problem_still_validates_and_keeps_its_declarations():
    original = heat_with_dirichlet()
    result = normalize(original)
    assert result.problem.partitions == original.partitions
    assert result.problem.time == original.time
    assert ons(normalize(result.problem).problem) == ons(result.problem)


def text(problem):
    return [str(c) for c in problem.constraints]


class TestCanonicalForms:
    def test_the_laplacian_becomes_div_grad(self):
        result = normalize(heat())
        assert "Δ" not in text(result.problem)[0]
        assert "∇_x·(∇_x w)" in text(result.problem)[0]

    def test_lap_and_div_grad_normalize_to_the_same_constraint(self):
        p = FunctionSymbol(Variable("p", dim=1), x)
        a = Problem((Eq(lap(p, x), 0).on(Omega),))
        b = Problem((Eq(div(grad(p, x), x), 0).on(Omega),))
        assert text(normalize(a).problem) == text(normalize(b).problem)

    def test_vector_laplacian_keeps_its_shape(self):
        v = FunctionSymbol(Variable("v", dim=2), x)
        problem = Problem((Eq(lap(v, x), 0).on(Omega),))
        assert normalize(problem).problem.constraints[0].lhs.op.name == "div"

    def test_both_sides_become_one_residual(self):
        p = FunctionSymbol(Variable("p", dim=1), x)
        f = FunctionSymbol(Variable("f", dim=1), x)
        result = normalize(Problem((Eq(-lap(p, x), f).on(Omega),)))
        constraint = result.problem.constraints[0]
        assert constraint.rhs == Const(0)
        assert constraint.lhs.op.name == "sub"
        assert result.origin[0].rhs == f

    def test_a_zero_right_side_is_kept(self):
        p = FunctionSymbol(Variable("p", dim=1), x)
        problem = Problem((Eq(p, 0).on(Omega),))
        assert normalize(problem).problem is problem

    def test_inequalities_become_residuals(self):
        p = FunctionSymbol(Variable("p", dim=1), x)
        problem = Problem((Le(p, 1).on(Omega),))
        constraint = normalize(problem).problem.constraints[0]
        assert constraint.rel is Relation.LE and constraint.rhs == Const(0)

    def test_complementarity_is_not_rewritten(self):
        p = FunctionSymbol(Variable("p", dim=1), x)
        q = FunctionSymbol(Variable("q", dim=1), x)
        problem = Problem((Complementary(p, q).on(Omega),))
        assert normalize(problem).problem is problem

    def test_normalizing_twice_changes_nothing(self):
        once = normalize(heat_with_dirichlet()).problem
        assert text(normalize(once).problem) == text(once)

    def test_the_input_problem_is_untouched(self):
        original = heat_with_dirichlet()
        before = text(original)
        normalize(original)
        assert text(original) == before

    def test_fixtures_still_validate_after_normalizing(self):
        for make in (poisson, heat, neumann_poisson, delay, periodic_poisson):
            assert errors(validate(normalize(make()).problem)) == ()
