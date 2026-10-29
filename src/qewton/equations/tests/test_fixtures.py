from qewton.equations import Problem
from qewton.equations.fixtures import heat, poisson


def test_poisson_builds_and_prints():
    p = poisson()
    assert isinstance(p, Problem)
    assert len(p.constraints) == 3
    assert {s.name for s in p.symbols()} == {"u", "f", "g", "h", "nu"}
    assert "Gamma_D" in str(p)


def test_heat_builds_and_prints():
    p = heat()
    assert len(p.constraints) == 2
    assert p.time is not None
    assert "∂w/∂t" in str(p)


def test_fixtures_build_fresh_symbols_each_call():
    # symbols compare by identity, so two calls give distinct but identically printed problems
    assert str(poisson()) == str(poisson())
    assert poisson().symbols().isdisjoint(poisson().symbols())


def test_neumann_poisson_has_a_global_constraint():
    from qewton.equations.fixtures import neumann_poisson

    p = neumann_poisson()
    assert [c.on is None for c in p.constraints] == [False, False, True]
    assert "globally" in str(p)


def test_delay_has_a_shift_and_a_history():
    from qewton.equations.expr.nodes import Shift, walk
    from qewton.equations.fixtures import delay

    p = delay()
    assert any(isinstance(n, Shift) for n in walk(p.constraints[0].rhs))
    assert "u(t − 1)" in str(p)


def test_periodic_poisson_declares_the_identification():
    from qewton.equations.fixtures import periodic_poisson

    p = periodic_poisson()
    assert len(p.periodic) == 1
    assert "Gamma_R = Gamma_L + (2, 0) along x" in str(p)
