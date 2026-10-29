import pytest

from qewton.equations import (
    Boundary,
    Domain,
    Eq,
    FunctionSymbol,
    Part,
    Partition,
    Periodic,
    Problem,
    Product,
    Restrict,
    Severity,
    Slice,
    Variable,
    dt,
    grad,
    lap,
    shift,
    validate,
)
from qewton.equations.analysis.diagnostics import Diagnostic, errors
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
dO = Boundary(Omega)
GD, GN = Part("Gamma_D", dO), Part("Gamma_N", dO)
u = FunctionSymbol(Variable("u", dim=1), x)
f = FunctionSymbol(Variable("f", dim=1), x)


def messages(problem, severity=None):
    return [
        d.message for d in validate(problem) if severity is None or d.severity is severity
    ]


def test_diagnostic_prints_severity_message_and_place():
    d = Diagnostic(Severity.WARNING, "careful", Omega)
    assert str(d) == "warning: careful  [Omega]"
    assert str(Diagnostic(Severity.ERROR, "bad")) == "error: bad"


class TestFixtures:
    @pytest.mark.parametrize(
        "make", [poisson, heat, neumann_poisson, delay, periodic_poisson]
    )
    def test_no_errors(self, make):
        assert errors(validate(make())) == ()

    def test_clean_problems_have_no_findings(self):
        assert validate(poisson()) == ()
        assert validate(periodic_poisson()) == ()
        assert validate(delay()) == ()

    def test_heat_only_warns_about_the_undeclared_unknown(self):
        assert [d.where.name for d in validate(heat())] == ["w"]


class TestPartitions:
    def problem(self, parts, whole=dO, held=(GD, GN)):
        return Problem(
            constraints=tuple(Eq(u, 0).on(part) for part in held),
            partitions=(Partition(whole, parts),),
        )

    def test_uncovered_part_is_warned(self):
        found = messages(self.problem((GD, GN), held=(GD,)), Severity.WARNING)
        assert found == ["No constraint holds on Gamma_N."]

    def test_part_of_another_set(self):
        other = Part("Other", Boundary(Domain("Q", ambient=x)))
        found = messages(self.problem((GD, other)), Severity.ERROR)
        assert len(found) == 1 and "not of" in found[0]

    def test_duplicate_and_empty_parts(self):
        assert "listed twice" in messages(self.problem((GD, GD)), Severity.ERROR)[0]
        assert "at least one" in messages(self.problem(()), Severity.ERROR)[0]


class TestPeriodic:
    L = (2.0, 0)
    left, right = Part("L", dO), Part("R", dO)

    def problem(self, offset, declared=None):
        return Problem(
            constraints=(
                Eq(
                    Restrict(u, self.right),
                    shift(Restrict(u, self.left), x, offset),
                ).on(self.right),
            ),
            periodic=(Periodic(self.left, self.right, x, declared or self.L),),
        )

    def test_matching_offset_is_clean(self):
        assert errors(validate(self.problem(self.L))) == ()
        assert errors(validate(self.problem((-2, 0)))) == ()

    def test_wrong_offset_is_an_error(self):
        found = messages(self.problem((3.0, 0)), Severity.ERROR)
        assert len(found) == 1 and "declared periodic" in found[0]

    def test_declaration_without_a_constraint_is_warned(self):
        problem = Problem(
            constraints=(Eq(u, 0).on(self.right),),
            periodic=(Periodic(self.left, self.right, x, self.L),),
        )
        assert "relates the two" in messages(problem, Severity.WARNING)[0]


class TestDelays:
    I = Domain("I", ambient=t)
    H = Domain("H", ambient=t)
    w = FunctionSymbol(Variable("w", dim=1), t)
    phi = FunctionSymbol(Variable("phi", dim=1), t)

    def test_history_on_an_interval_is_enough(self):
        assert validate(delay()) == ()

    def test_missing_history(self):
        problem = Problem((Eq(dt(self.w, t), shift(self.w, t, -1)).on(self.I),), time=t)
        assert "needs history" in messages(problem, Severity.WARNING)[0]

    def test_a_slice_is_not_history(self):
        start = Slice(t, 0.0)
        problem = Problem(
            (
                Eq(dt(self.w, t), shift(self.w, t, -1)).on(self.I),
                Eq(Restrict(self.w, start), 1).on(start),
            ),
            time=t,
        )
        found = messages(problem, Severity.WARNING)
        assert any("not enough for a delay" in m for m in found)

    def test_advance_is_flagged(self):
        problem = Problem((Eq(dt(self.w, t), shift(self.w, t, 1)).on(self.I),), time=t)
        assert any("advance" in m for m in messages(problem))

    def test_symbolic_delay_counts_as_a_delay(self):
        tau = FunctionSymbol(Variable("tau", dim=1), None)
        problem = Problem((Eq(dt(self.w, t), shift(self.w, t, -tau)).on(self.I),), time=t)
        assert "needs history" in messages(problem, Severity.WARNING)[0]

    def test_no_check_without_a_declared_time(self):
        problem = Problem((Eq(dt(self.w, t), shift(self.w, t, -1)).on(self.I),))
        assert validate(problem) == ()


class TestRestrictions:
    def test_restricting_to_a_subdomain_is_fine(self):
        # a restriction need not be lower-dimensional: the history of a delay is one
        assert validate(Problem((Eq(Restrict(u, Omega), 0).on(Omega),))) == ()

    def test_restricting_a_function_that_lacks_the_axis(self):
        c = FunctionSymbol(Variable("c", dim=1), None)
        problem = Problem((Eq(Restrict(c, dO), 0).on(dO),))
        assert "does not depend on x" in messages(problem, Severity.WARNING)[0]

    def test_a_boundary_trace_is_fine(self):
        assert validate(Problem((Eq(Restrict(u, dO), 0).on(dO),))) == ()

    def test_time_slice_is_fine(self):
        w = FunctionSymbol(Variable("w", dim=1), x * t)
        start = Product((Omega, Slice(t, 0.0)))
        problem = Problem((Eq(Restrict(w, start), 0).on(start),), time=t)
        assert validate(problem) == ()


class TestShapes:
    def test_conflicts_are_errors(self):
        a = FunctionSymbol(Variable("a"), x)
        b = FunctionSymbol(Variable("b", dim=1), x)
        c = FunctionSymbol(Variable("c", dim=2), x)
        problem = Problem((Eq(lap(a, x), b).on(Omega), Eq(grad(a, x), c).on(Omega)))
        assert errors(validate(problem)) == ()
        bad = Problem((Eq(lap(a, x), b).on(Omega), Eq(a, c).on(Omega)))
        assert len(errors(validate(bad))) == 1

    def test_errors_come_first(self):
        a = FunctionSymbol(Variable("a"), x)
        b = FunctionSymbol(Variable("b"), x)
        c = FunctionSymbol(Variable("c", dim=2), x)
        d = FunctionSymbol(Variable("d", dim=1), x)
        problem = Problem((Eq(b, 0).on(Omega), Eq(a, c).on(Omega), Eq(a, d).on(Omega)))
        found = validate(problem)
        assert found[0].severity is Severity.ERROR
