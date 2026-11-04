import numpy as np

from qewton.config.variables import Variable
from qewton.equations import (
    BatchMode,
    Boundary,
    Domain,
    Eq,
    FunctionSymbol,
    Normal,
    Part,
    Partition,
    Problem,
    Restrict,
    batching,
    div,
    dot,
    grad,
    report,
    run_pipeline,
)


def scalar(name, inputs):
    return FunctionSymbol(Variable(name, dim=1), inputs)


def poisson(nu_on_x: bool = False):
    """The worked example of the plan, with nu constant in x unless `nu_on_x`."""
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    boundary = Boundary(omega)
    gd, gn = Part("Gamma_D", boundary), Part("Gamma_N", boundary)
    u, f, g, h = scalar("u", x), scalar("f", None), scalar("g", x), scalar("h", x)
    nu = scalar("nu", x if nu_on_x else None)
    problem = Problem(
        constraints=(
            Eq(-div(nu * grad(u, x), x), f).on(omega),
            Eq(Restrict(u, gd), g).on(gd),
            Eq(dot(Restrict(nu * grad(u, x), gn), Normal(boundary)), h).on(gn),
        ),
        partitions=(Partition(boundary, (gd, gn)),),
    )
    symbols = {"u": u, "f": f, "g": g, "h": h, "nu": nu, "omega": omega}
    return problem, symbols


def fixed_data(s, *without):
    data = {s["f"]: 1.0, s["nu"]: 1.0, s["g"]: 0.0, s["h"]: 0.0}
    return {k: v for k, v in data.items() if k not in without}


def entry_of(task, name):
    return next(e for e in batching(task).entries if e.owner.variable.name == name)


def test_a_constant_coefficient_is_affine_in_the_operator():
    problem, s = poisson()
    task = problem.task(
        unknown=[s["u"]],
        fixed=fixed_data(s, s["nu"]),
        varying={s["nu"]: np.linspace(1, 2, 5)},
    )
    entry = entry_of(task, "nu")
    assert entry.mode is BatchMode.OPERATOR_AFFINE
    assert entry.constraints == (0, 2)
    assert [u.variable.name for u in entry.reaches] == ["u"]
    assert entry.axis.name == "p_nu"


def test_a_coefficient_that_depends_on_x_is_not_separable():
    problem, s = poisson(nu_on_x=True)
    task = problem.task(
        unknown=[s["u"]],
        fixed={**fixed_data(s, s["nu"]), s["omega"]: object()},
        varying={s["nu"]: np.ones((3, 4))},
    )
    assert entry_of(task, "nu").mode is BatchMode.OPERATOR_NONLINEAR


def test_a_varying_source_only_changes_the_right_hand_side():
    problem, s = poisson()
    task = problem.task(
        unknown=[s["u"]], fixed=fixed_data(s, s["f"]), varying={s["f"]: np.ones(4)}
    )
    entry = entry_of(task, "f")
    assert entry.mode is BatchMode.SOURCE_ONLY
    assert entry.constraints == (0,)


def test_a_varying_set_is_a_geometry_batch():
    problem, s = poisson()
    task = problem.task(
        unknown=[s["u"]], fixed=fixed_data(s), varying={s["omega"]: "moving"}
    )
    (entry,) = batching(task).entries
    assert entry.mode is BatchMode.GEOMETRY
    assert entry.owner == s["omega"]
    assert [u.variable.name for u in entry.reaches] == ["u"]


def test_a_task_without_varying_data_has_no_batching():
    problem, s = poisson()
    task = problem.task(unknown=[s["u"]], fixed=fixed_data(s))
    assert not batching(task)
    assert str(batching(task)) == "no batch axes"


def test_report_has_a_batching_section_only_for_batched_tasks():
    problem, s = poisson()
    plain = problem.task(unknown=[s["u"]], fixed=fixed_data(s))
    assert "Batching" not in report(run_pipeline(plain))
    batched = problem.task(
        unknown=[s["u"]], fixed=fixed_data(s, s["f"]), varying={s["f"]: np.ones(3)}
    )
    text = report(run_pipeline(batched))
    assert "p_f (f): source only in C1; reaches u" in text
    assert "parallel over the batch" in text
