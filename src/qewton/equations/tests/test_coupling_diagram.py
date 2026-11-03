import matplotlib

matplotlib.use("Agg")

from qewton.equations import coupling_diagram, plot_coupling
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
