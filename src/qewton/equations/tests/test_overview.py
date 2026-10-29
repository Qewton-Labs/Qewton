from qewton.equations import overview
from qewton.equations.fixtures import heat, poisson


def names(items):
    return [str(i) for i in items]


def test_poisson_overview():
    o = poisson().overview()
    assert o.time is None
    assert [a.name for a in o.space] == ["x"]
    assert set(names(o.sets)) == {"Omega", "∂Omega", "Gamma_D", "Gamma_N"}
    assert len(o.functions) == 5 and o.constraints == 3


def test_heat_lists_time_and_the_factors_of_products():
    o = heat().overview()
    assert o.time.name == "t"
    assert [a.name for a in o.space] == ["x"]
    assert {"Omega", "I", "Omega × I"} <= set(names(o.sets))


def test_assumed_shapes_are_marked():
    assert "w(x, t)  scalar (assumed)" in str(heat().overview())
    assert "(assumed)" not in str(poisson().overview())


def test_functions_show_their_inputs():
    text = str(poisson().overview())
    assert "u(x)  scalar" in text
    assert "nu  (constant)  scalar" in text


def test_printed_overview_has_all_sections():
    text = str(overview(poisson()))
    for header in ("coordinates:", "sets:", "functions:", "constraints: 3"):
        assert header in text
    assert "x  space, dim 2" in text
    assert "boundary of Omega" in text
