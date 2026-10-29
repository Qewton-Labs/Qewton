import pytest

from qewton.equations import (
    AnalysisContext,
    Domain,
    Eq,
    FunctionSymbol,
    Problem,
    Variable,
    report,
    run_pipeline,
)
from qewton.equations.analysis.diagnostics import Severity
from qewton.equations.fixtures import (
    delay,
    heat,
    neumann_poisson,
    periodic_poisson,
    poisson,
)


def named(problem, name):
    return next(s for s in problem.symbols() if s.variable.name == name)


def forward_poisson():
    p = poisson()
    fixed = {named(p, n): 1.0 for n in ("f", "g", "h", "nu")}
    return p.task(unknown=[named(p, "u")], fixed=fixed)


GOLDEN_POISSON = """\
Task
  unknown: u
  fixed: f, g, h, nu
Signatures
  f : · → ℝ (assumed)  [fixed]  on Omega
  g : x → ℝ (assumed)  [fixed]  on Gamma_D
  h : x → ℝ (assumed)  [fixed]  on Gamma_N
  nu : · → ℝ  [fixed]  on Omega, Gamma_N
  u : x → ℝ  [unknown]  on Omega, Gamma_D, Gamma_N
Normalized
  constraints:
    1. −∇_x·(nu·∇_x u) − f = 0 on Omega
    2. u|_{Gamma_D} − g = 0 on Gamma_D
    3. dot((nu·∇_x u)|_{Gamma_N}, n_{∂Omega}) − h = 0 on Gamma_N
  partitions:
    ∂Omega = Gamma_D ⊔ Gamma_N
Diagnostics
  none"""


def test_golden_report_for_poisson():
    assert report(run_pipeline(forward_poisson())) == GOLDEN_POISSON


GOLDEN_DELAY_SIGNATURES = """\
Signatures
  phi : t → ℝ  [fixed]  on H
  u : t → ℝ  [unknown]  on H, I"""


def test_golden_signatures_for_the_delay_problem():
    p = delay()
    task = p.task(unknown=[named(p, "u")], fixed={named(p, "phi"): 1.0})
    text = report(run_pipeline(task))
    assert GOLDEN_DELAY_SIGNATURES in text
    assert text.endswith("Diagnostics\n  none")


def test_the_context_keeps_the_problem_and_its_normalized_form():
    task = forward_poisson()
    ctx = run_pipeline(task)
    assert isinstance(ctx, AnalysisContext)
    assert ctx.task is task
    assert ctx.origin == task.problem.constraints
    assert "Δ" not in str(ctx.problem)
    assert str(task.problem) != str(ctx.problem)


def test_findings_of_every_pass_are_collected_errors_first():
    p = heat()
    ctx = run_pipeline(p.task(unknown=[named(p, "w")]))
    kinds = [d.severity for d in ctx.diagnostics]
    assert kinds == sorted(kinds, key=lambda k: list(Severity).index(k))
    messages = " ".join(d.message for d in ctx.diagnostics)
    assert "neither unknown nor given a value" in messages
    assert "not determined" in messages


def test_a_lift_shows_up_as_information():
    x, t = Variable("x", dim=2), Variable("t", dim=1)
    from qewton.equations import Boundary, Product, Restrict, dt

    omega, interval = Domain("Omega", ambient=x), Domain("I", ambient=t)
    boundary = Boundary(omega)
    w = FunctionSymbol(Variable("w", dim=1), x * t)
    g = FunctionSymbol(Variable("g", dim=1), x)
    problem = Problem(
        (
            Eq(dt(w, t), 0).on(Product((omega, interval))),
            Eq(Restrict(w, boundary), g).on(boundary),
        ),
        time=t,
    )
    ctx = run_pipeline(problem.task(unknown=[w], fixed={g: 0.0}))
    assert [d.severity for d in ctx.diagnostics] == [Severity.INFO]
    assert ctx.ok


@pytest.mark.parametrize(
    "make", [poisson, heat, neumann_poisson, delay, periodic_poisson]
)
def test_every_fixture_runs_through_the_pipeline(make):
    ctx = run_pipeline(make().task())
    assert ctx.ok
    assert len(ctx.signatures.table) == len(ctx.task.problem.symbols())
    assert report(ctx).startswith("Task")


def test_a_problem_with_a_validation_error_is_not_ok():
    from qewton.equations import Boundary, Part, Partition

    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    boundary = Boundary(omega)
    part = Part("G", boundary)
    u = FunctionSymbol(Variable("u", dim=1), x)
    problem = Problem(
        (Eq(u, 0).on(part),), partitions=(Partition(boundary, (part, part)),)
    )
    ctx = run_pipeline(problem.task(unknown=[u]))
    assert not ctx.ok
    assert ctx.diagnostics[0].severity is Severity.ERROR
