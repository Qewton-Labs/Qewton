import pytest

from qewton.config.variables import Variable
from qewton.equations import (
    Boundary,
    ConditionKind,
    Domain,
    Eq,
    FunctionSymbol,
    Interface,
    Normal,
    Part,
    Partition,
    PdeType,
    Problem,
    classify,
    jump,
    lap,
    run_pipeline,
    trace,
)
from qewton.equations.analysis.prepare.validate import validate
from qewton.equations.fixtures import transmission


def unknown_task(problem, name="u"):
    return problem.task(unknown=[s for s in problem.symbols() if s.name == name])


def test_interface_needs_two_different_parts_of_one_domain():
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    other = Domain("Other", ambient=x)
    a, b = Part("A", omega), Part("B", omega)
    assert Interface(a, b).codim == 1
    assert str(Interface(a, b)) == "A | B"
    with pytest.raises(ValueError):
        Interface(a, a)
    with pytest.raises(ValueError):
        Interface(a, Part("C", other))
    with pytest.raises(TypeError):
        Interface(omega, a)  # type: ignore[arg-type]


def test_normal_exists_on_an_interface():
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    gamma = Interface(Part("A", omega), Part("B", omega))
    assert Normal(gamma).of == gamma
    with pytest.raises(ValueError):
        Normal(omega)


def test_trace_and_jump_print_and_check_the_side():
    problem = transmission()
    gamma = problem.interfaces[0]
    u = next(s for s in problem.symbols() if s.name == "u")
    assert str(trace(u, gamma, 0)) == "u|_{Omega_1}|_{Omega_1 | Omega_2}"
    assert "−" in str(jump(u, gamma))
    with pytest.raises(ValueError):
        trace(u, gamma, 2)


def test_transmission_problem_is_valid_and_prints_its_interface():
    problem = transmission()
    assert run_pipeline(unknown_task(problem)).ok
    assert "interfaces:\n  Omega_1 | Omega_2" in str(problem)


def test_transmission_is_classified_with_interface_conditions():
    result = classify(unknown_task(transmission()))
    assert result.pde_types == {PdeType.ELLIPTIC}
    assert [f.value for f in result.conditions] == [
        ConditionKind.INTERFACE_VALUE,
        ConditionKind.INTERFACE_FLUX,
        ConditionKind.DIRICHLET,
    ]
    interface = [f for f in result.counts if f.rule == "interface_count"]
    assert [f.value for f in interface] == [(1, 1), (1, 1)]
    assert not [d for d in result.diagnostics if d.severity.value == "warning"]


def test_missing_flux_condition_is_a_warning():
    problem = transmission()
    kept = tuple(c for i, c in enumerate(problem.constraints) if i != 3)
    broken = Problem(kept, problem.partitions, interfaces=problem.interfaces)
    result = classify(unknown_task(broken))
    messages = [str(d) for d in result.diagnostics]
    assert any("interface flux" in m and "0 given, 1 needed" in m for m in messages)


def test_undeclared_interface_is_reported_by_validate():
    problem = transmission()
    undeclared = Problem(problem.constraints, problem.partitions)
    assert any("not declared" in str(d) for d in validate(undeclared))
    assert not any("not declared" in str(d) for d in validate(problem))


def test_interface_outside_a_partition_is_reported():
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    a, b = Part("A", omega), Part("B", omega)
    u = FunctionSymbol(Variable("u", dim=1), x)
    problem = Problem(
        (Eq(-lap(u, x), 0).on(a), Eq(u, 0).on(Boundary(omega))),
        interfaces=(Interface(a, b),),
    )
    assert any("partition" in str(d) for d in validate(problem))
