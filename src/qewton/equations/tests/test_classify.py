import pytest

from qewton.equations import fixtures
from qewton.equations.analysis.context import classify
from qewton.equations.analysis.classify.conditions import ConditionKind
from qewton.equations.analysis.classify.equations import PdeType, pde_type
from qewton.equations.analysis.classify.spaces import FunctionSpace

UNKNOWNS = {
    "poisson": ("u",),
    "heat": ("w",),
    "wave": ("w",),
    "advection": ("u",),
    "obstacle": ("u",),
    "delay": ("u",),
    "neumann_poisson": ("p",),
    "periodic_poisson": ("u",),
    "stokes": ("u", "p"),
}


def run(name):
    problem = getattr(fixtures, name)()
    unknown = [s for s in problem.symbols() if s.variable.name in UNKNOWNS[name]]
    return classify(problem.task(unknown=unknown))


def kinds(facts):
    return [f.value for f in facts]


def spaces(result):
    return {str(s.variable.name): v for s, v in result.space_of.items()}


@pytest.mark.parametrize(
    "name, expected",
    [
        ("poisson", PdeType.ELLIPTIC),
        ("heat", PdeType.PARABOLIC),
        ("wave", PdeType.HYPERBOLIC_WAVE),
        ("advection", PdeType.HYPERBOLIC_TRANSPORT),
        ("obstacle", PdeType.ELLIPTIC),
        ("delay", PdeType.ODE),
        ("neumann_poisson", PdeType.ELLIPTIC),
        ("periodic_poisson", PdeType.ELLIPTIC),
        ("stokes", PdeType.SADDLE_POINT),
    ],
)
def test_block_types(name, expected):
    assert run(name).pde_types == {expected}


def test_pde_type_table():
    assert pde_type(0, 0) is PdeType.ALGEBRAIC
    assert pde_type(3, 0) is PdeType.ODE
    assert pde_type(0, 4) is PdeType.UNCLASSIFIED


def test_stokes_equations_and_multiplier():
    result = run("stokes")
    assert kinds(result.equations) == [PdeType.ELLIPTIC, PdeType.FIRST_ORDER_STATIONARY]
    assert "p only appears" in result.blocks[0].why


def test_obstacle_has_a_relation_fact():
    result = run("obstacle")
    assert [f.rule for f in result.equations] == ["pde_type"]
    assert [f.rule for f in result.relations] == ["relation"]


def test_condition_kinds():
    assert kinds(run("poisson").conditions)[:2] == [
        ConditionKind.DIRICHLET,
        ConditionKind.NEUMANN,
    ]
    assert set(kinds(run("heat").conditions)) == {ConditionKind.INITIAL}
    assert kinds(run("wave").conditions) == [
        ConditionKind.DIRICHLET,
        ConditionKind.INITIAL,
        ConditionKind.INITIAL,
    ]
    assert kinds(run("periodic_poisson").conditions) == [ConditionKind.PERIODIC]
    assert ConditionKind.GLOBAL in kinds(run("neumann_poisson").conditions)
    assert kinds(run("delay").conditions) == [ConditionKind.INITIAL]


def test_neumann_condition_is_neumann():
    assert ConditionKind.NEUMANN in kinds(run("neumann_poisson").conditions)


@pytest.mark.parametrize("name", ["poisson", "wave", "delay", "periodic_poisson"])
def test_complete_problems_have_balanced_counts(name):
    result = run(name)
    assert result.counts
    assert all(f.value[0] == f.value[1] for f in result.counts)
    assert not [d for d in result.diagnostics if d.severity.value == "warning"]


def test_wave_needs_two_initial_conditions():
    initial = [f for f in run("wave").counts if f.rule == "initial_count"]
    assert [f.subject for f in initial] == ["w, order 0", "w, order 1"]


def test_stokes_warns_about_missing_boundary_conditions():
    messages = [str(d) for d in run("stokes").diagnostics]
    assert any("No boundary conditions" in m for m in messages)


def test_advection_notes_the_inflow():
    messages = [str(d) for d in run("advection").diagnostics]
    assert any("flow enters" in m for m in messages)


def test_function_spaces():
    assert spaces(run("stokes")) == {"u": FunctionSpace.H1, "p": FunctionSpace.L2}
    for name in ("poisson", "wave", "advection"):
        assert set(spaces(run(name)).values()) == {FunctionSpace.H1}


def test_report_prints_every_section():
    text = str(run("stokes"))
    for title in ("equations:", "blocks:", "function spaces:"):
        assert title in text


def test_equations_on_subdomains_are_bulk_equations():
    from qewton.config.variables import Variable
    from qewton.equations import (
        Boundary,
        Domain,
        Eq,
        FunctionSymbol,
        Part,
        Partition,
        Problem,
        lap,
    )

    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    inner, outer = Part("Inner", omega), Part("Outer", omega)
    u = FunctionSymbol(Variable("u", dim=1), x)
    f = FunctionSymbol(Variable("f", dim=1), x)
    g = FunctionSymbol(Variable("g", dim=1), x)
    problem = Problem(
        constraints=(
            Eq(-lap(u, x), f).on(inner),
            Eq(u - lap(u, x), f).on(outer),
            Eq(u, g).on(Boundary(omega)),
        ),
        partitions=(Partition(omega, (inner, outer)),),
    )
    result = classify(problem.task(unknown=[u]))
    assert kinds(result.equations) == [PdeType.ELLIPTIC] * 2
    assert kinds(result.conditions) == [ConditionKind.DIRICHLET]
    assert [f.value for f in result.counts] == [(1, 1)]
    assert spaces(result) == {"u": FunctionSpace.H1}
