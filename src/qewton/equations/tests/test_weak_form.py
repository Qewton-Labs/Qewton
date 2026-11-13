from qewton.config.variables import Variable
from qewton.equations import (
    Boundary,
    Domain,
    Eq,
    FunctionSpace,
    FunctionSymbol,
    Normal,
    Part,
    Partition,
    Problem,
    Restrict,
    div,
    dot,
    fixtures,
    grad,
    weak_form,
)
from qewton.equations.diagnostics import Severity
from qewton.equations.analysis.structure.structure import signed_terms
from qewton.equations.expr.ops import Linearity


def weak_of(name, *unknown):
    problem = getattr(fixtures, name)()
    symbols = [s for s in problem.symbols() if s.variable.name in unknown]
    return weak_form(problem.task(unknown=symbols))


def residual_text(weak, test):
    found = [r for r in weak.residuals if r.test.variable.name == test]
    return str(found[0])


def severities(weak):
    return [d.severity for d in weak.diagnostics]


class TestPoisson:
    def test_form(self):
        weak = weak_of("poisson", "u")
        assert residual_text(weak, "v_u") == (
            "F[v_u] = + ∫ inner(nu·∇_x u, ∇_x v_u) dx - ∫ inner(f, v_u) dx"
            " - ∫ inner(h, v_u) ds(Gamma_N) = 0"
        )

    def test_dirichlet_is_essential_and_not_in_the_form(self):
        weak = weak_of("poisson", "u")
        assert len(weak.essential) == 1
        assert "Gamma_D" not in residual_text(weak, "v_u")

    def test_space_and_no_diagnostics(self):
        weak = weak_of("poisson", "u")
        (u,) = weak.unknowns
        assert weak.spaces[u] is FunctionSpace.H1
        assert weak.diagnostics == ()

    def test_test_function_follows_the_unknown(self):
        weak = weak_of("poisson", "u")
        (u,) = weak.unknowns
        v = weak.tests[u]
        assert v.variable.name == "v_u"
        assert v.shape == u.shape
        assert v.inputs is u.inputs

    def test_operator_and_source_split(self):
        weak = weak_of("poisson", "u")
        residual = weak.residuals[0]
        assert [str(t.measure) for t in weak.operator_terms(residual)] == ["dx"]
        assert [str(t.measure) for t in weak.source_terms(residual)] == [
            "dx",
            "ds(Gamma_N)",
        ]
        assert weak.linearity is Linearity.LINEAR


class TestHeat:
    def test_time_derivative_is_a_mass_term(self):
        weak = weak_of("heat", "w")
        text = residual_text(weak, "v_w")
        assert text.startswith(
            "F[v_w] = + ∫ inner(∂w/∂t, v_w) dx + ∫ inner(∇_x w, ∇_x v_w) dx"
        )

    def test_test_function_has_no_time_input(self):
        weak = weak_of("heat", "w")
        (w,) = weak.unknowns
        assert [a.name for a in weak.tests[w].coordinates()] == ["x"]

    def test_initial_condition_is_kept(self):
        weak = weak_of("heat", "w")
        assert len(weak.constraints) == 1
        assert weak.time is not None

    def test_boundary_without_condition_is_kept_with_a_warning(self):
        weak = weak_of("heat", "w")
        assert "ds" in residual_text(weak, "v_w")
        assert Severity.WARNING in severities(weak)


class TestStokes:
    def test_momentum_and_continuity(self):
        weak = weak_of("stokes", "u", "p")
        momentum = residual_text(weak, "v_u")
        assert "∫ inner(∇_x u, ∇_x v_u) dx" in momentum
        assert "- ∫ inner(p, ∇_x·v_u) dx" in momentum
        assert residual_text(weak, "v_p") == "F[v_p] = + ∫ inner(∇_x·u, v_p) dx = 0"

    def test_spaces(self):
        weak = weak_of("stokes", "u", "p")
        spaces = {u.variable.name: s for u, s in weak.spaces.items()}
        assert spaces == {"u": FunctionSpace.H1, "p": FunctionSpace.L2}

    def test_pairing_uses_the_free_unknown_for_continuity(self):
        weak = weak_of("stokes", "u", "p")
        paired = {r.row: r.unknown.variable.name for r in weak.residuals}
        assert paired == {0: "u", 1: "p"}


class TestOtherProblems:
    def test_neumann_on_the_whole_boundary(self):
        weak = weak_of("neumann_poisson", "p")
        assert residual_text(weak, "v_p") == (
            "F[v_p] = + ∫ inner(∇_x p, ∇_x v_p) dx - ∫ inner(f, v_p) dx"
            " - ∫ inner(h, v_p) ds = 0"
        )
        assert len(weak.constraints) == 1  # the zero mean stays a constraint

    def test_periodic_parts_cancel(self):
        weak = weak_of("periodic_poisson", "u")
        assert "ds" not in residual_text(weak, "v_u")
        assert len(weak.periodic) == 1
        assert not weak.essential

    def test_transport_is_not_integrated_by_parts(self):
        weak = weak_of("advection", "u")
        assert "ds" not in residual_text(weak, "v_u")
        assert len(weak.essential) == 1  # the inflow condition

    def test_constant_coefficient_stays_outside_the_test_function(self):
        weak = weak_of("wave", "w")
        assert "c²·∇_x v_w" in residual_text(weak, "v_w")

    def test_subdomain_equations_share_a_test_function(self):
        weak = weak_of("transmission", "u")
        assert [r.test.variable.name for r in weak.residuals] == ["v_u", "v_u"]
        assert {str(t.measure) for r in weak.residuals for t in r.terms} == {
            "dx(Omega_1)",
            "dx(Omega_2)",
        }

    def test_what_is_not_compiled_is_an_error(self):
        interface = weak_of("transmission", "u")
        assert any("Interface" in d.message for d in interface.diagnostics)
        obstacle = weak_of("obstacle", "u")
        assert any("Inequalities" in d.message for d in obstacle.diagnostics)
        assert Severity.ERROR in severities(obstacle)


def poisson_with(natural):
    """-div(nu grad u) = f; `natural` builds the condition on Gamma_N from flux, u."""
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    d_omega = Boundary(omega)
    gamma_d, gamma_n = Part("Gamma_D", d_omega), Part("Gamma_N", d_omega)
    u = FunctionSymbol(Variable("u", dim=1), x)
    f = FunctionSymbol(Variable("f"), x)
    h = FunctionSymbol(Variable("h"), x)
    alpha = FunctionSymbol(Variable("alpha"), None)
    nu = FunctionSymbol(Variable("nu", dim=1), None)
    flux = nu * grad(u, x)
    on_n = dot(Restrict(flux, gamma_n), Normal(d_omega))
    constraints = (
        Eq(-div(flux, x), f).on(omega),
        Eq(Restrict(u, gamma_d), 0).on(gamma_d),
        natural(on_n, Restrict(u, gamma_n), alpha, h).on(gamma_n),
    )
    problem = Problem(
        constraints=constraints, partitions=(Partition(d_omega, (gamma_d, gamma_n)),)
    )
    return weak_form(problem.task(unknown=[u]))


class TestBoundaryTerms:
    def test_robin_substitutes_the_flux(self):
        weak = poisson_with(lambda flux, u, alpha, h: Eq(flux + alpha * u, h))
        text = residual_text(weak, "v_u")
        assert "+ ∫ inner(alpha·u|_{Gamma_N}, v_u) ds(Gamma_N)" in text
        assert "- ∫ inner(h, v_u) ds(Gamma_N)" in text
        assert weak.diagnostics == ()

    def test_homogeneous_neumann_adds_no_term(self):
        weak = poisson_with(lambda flux, u, alpha, h: Eq(flux, 0))
        assert "ds" not in residual_text(weak, "v_u")
        assert weak.diagnostics == ()

    def test_flux_written_with_the_other_sign(self):
        weak = poisson_with(lambda flux, u, alpha, h: Eq(h, flux))
        assert "- ∫ inner(h, v_u) ds(Gamma_N)" in residual_text(weak, "v_u")

    def test_condition_that_is_not_the_flux_keeps_the_raw_term(self):
        def bare_derivative(flux, u, alpha, h):
            x = Variable("x", dim=2)
            normal = Normal(Boundary(Domain("Omega", ambient=x)))
            return Eq(dot(Restrict(grad(u.expr, x), u.on), normal), h)

        weak = poisson_with(bare_derivative)
        assert Severity.WARNING in severities(weak)
        assert "dot((nu·∇_x u)|_{Gamma_N}, n_{∂Omega})" in residual_text(weak, "v_u")

    def test_dirichlet_on_both_parts(self):
        weak = poisson_with(lambda flux, u, alpha, h: Eq(u, h))
        assert Severity.WARNING not in severities(weak)
        assert len(weak.essential) == 2
        assert "ds" not in residual_text(weak, "v_u")


class TestPairing:
    def test_equation_without_a_partner_is_an_error(self):
        x = Variable("x", dim=2)
        omega = Domain("Omega", ambient=x)
        p = FunctionSymbol(Variable("p", dim=1), x)
        # Two scalar equations, but only one scalar unknown: one has no partner.
        problem = Problem(
            constraints=(
                Eq(div(grad(p, x), x), 0).on(omega),
                Eq(p, 0).on(omega),
            )
        )
        weak = weak_form(problem.task(unknown=[p]))
        assert Severity.ERROR in severities(weak)
        assert len(weak.residuals) == 1

    def test_every_row_gets_its_own_unknown(self):
        weak = weak_of("stokes", "u", "p")
        assert len({r.unknown for r in weak.residuals}) == 2


def test_signed_terms_is_public():
    x = Variable("x", dim=1)
    u = FunctionSymbol(Variable("u"), x)
    assert [s for _, s in signed_terms(u - u + u)] == [1, -1, 1]
