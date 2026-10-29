import pytest

from qewton.config.variables import Variable
from qewton.equations import (
    Boundary,
    Domain,
    Eq,
    FunctionSymbol,
    Part,
    Periodic,
    Problem,
    Restrict,
    lap,
    shift,
)

x = Variable("x", dim=2)
t = Variable("t", dim=1)
Omega = Domain("Omega", ambient=x)
dO = Boundary(Omega)
GL, GR = Part("Gamma_L", dO), Part("Gamma_R", dO)
u = FunctionSymbol(Variable("u", dim=1), x)


class TestPeriodic:
    def test_builds_and_prints(self):
        p = Periodic(GL, GR, x, (2.0, 0))
        assert str(p) == "Gamma_R = Gamma_L + (2, 0) along x"

    def test_equal_by_value_and_list_offset_becomes_tuple(self):
        assert Periodic(GL, GR, x, [2, 0]) == Periodic(GL, GR, x, (2, 0))

    def test_one_dimensional_axis_takes_a_number(self):
        a, b = Domain("A", ambient=t), Domain("B", ambient=t)
        assert Periodic(a, b, t, 1.0).offset == 1.0
        with pytest.raises(ValueError):
            Periodic(a, b, t, (1.0,))

    def test_offset_length_must_match_the_axis(self):
        with pytest.raises(ValueError):
            Periodic(GL, GR, x, (1, 0, 0))
        with pytest.raises(ValueError):
            Periodic(GL, GR, x, 1)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError, match="itself"):
            Periodic(GL, GL, x, (1, 0))
        with pytest.raises(ValueError, match="zero"):
            Periodic(GL, GR, x, (0, 0))
        with pytest.raises(TypeError):
            Periodic(GL, GR, x, ("a", 0))
        with pytest.raises(ValueError, match="same"):
            Periodic(GL, Omega, x, (1, 0))
        with pytest.raises(TypeError):
            Periodic("GL", GR, x, (1, 0))
        with pytest.raises(ValueError, match="not a coordinate"):
            Periodic(GL, GR, Variable("y", dim=2), (1, 0))


class TestProblemWithPeriodic:
    def problem(self):
        return Problem(
            constraints=(
                Eq(-lap(u, x), 1).on(Omega),
                Eq(Restrict(u, GR), shift(Restrict(u, GL), x, (2.0, 0))).on(GR),
            ),
            periodic=[Periodic(GL, GR, x, (2.0, 0))],
        )

    def test_prints_the_declaration(self):
        text = str(self.problem())
        assert "periodic:" in text
        assert "Gamma_R = Gamma_L + (2, 0) along x" in text

    def test_sets_include_both_sides(self):
        assert {GL, GR} <= self.problem().sets()

    def test_declarations_are_checked_and_tuples(self):
        assert isinstance(self.problem().periodic, tuple)
        with pytest.raises(TypeError):
            Problem(constraints=(Eq(u, 0).on(Omega),), periodic=("GL",))
