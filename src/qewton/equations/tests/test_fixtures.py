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
