import matplotlib

matplotlib.use("Agg")

from qewton.equations import (
    Domain,
    Eq,
    FunctionSymbol,
    Problem,
    Slice,
    Variable,
    coupling_diagram,
    dt,
    lap,
    plot_coupling,
)
from qewton.equations.fixtures import heat, stokes


def _named(problem, name):
    return next(s for s in problem.symbols() if s.variable.name == name)


def test_stokes_diagram_has_data_sets_unknowns_and_coordinates():
    problem = stokes()
    task = problem.task(unknown=[_named(problem, "u"), _named(problem, "p")])
    diagram = coupling_diagram(task)

    assert diagram.columns["equations"] == ("c1", "c2")
    assert diagram.columns["unknowns"] == ("p[0]", "u[0]", "u[1]")
    assert diagram.roles[("data", "f")] == "fixed"
    assert diagram.columns["sets"]
    assert diagram.columns["coordinates"]
    assert {e.kind for e in diagram.edges} >= {
        "occurs",
        "posed",
        "contains",
        "on",
        "in",
    }
    assert sum(e.matched for e in diagram.edges) >= 3


def test_heat_diagram_marks_varying_data_and_can_be_drawn():
    problem = heat()
    task = problem.task(unknown=[_named(problem, "w")])
    diagram = coupling_diagram(task)

    assert "t" in diagram.columns["coordinates"]
    assert plot_coupling(task) is not None


def test_data_and_unknowns_are_linked_to_every_set_they_are_evaluated_on():
    x, t = Variable("x", dim=2), Variable("t", dim=1)
    u = FunctionSymbol(Variable("u", dim=1), x * t)
    f = FunctionSymbol(Variable("f", dim=1), t)
    omega, interval = Domain("Omega", ambient=x), Domain("I", ambient=t)
    bulk, start = omega * interval, omega * Slice(t, 0.0)
    problem = Problem((Eq(dt(u, t) - lap(u, x), f).on(bulk), Eq(u, 0).on(start)), time=t)
    diagram = coupling_diagram(problem.task(unknown=[u], fixed=[f]))
    on = {(e.left[1], e.right[1]) for e in diagram.edges if e.kind == "on"}
    assert ("u[0]", str(start)) in on and ("u[0]", str(bulk)) in on
    assert ("f", str(bulk)) in on
