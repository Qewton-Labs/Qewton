"""Tests for coordinates and sets."""

# pylint: disable=invalid-name  # module names mirror the plan's notation: Omega, Gamma_D
import pytest

from qewton.config.variables import Variable
from qewton.equations.geometry.coordinates import check_coordinate
from qewton.geometries.base import Geometry
from qewton.geometries.continuous.domains_1d.interval import Interval as IntervalGeometry
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.equations.geometry.sets import (
    Boundary,
    Domain,
    Part,
    Product,
    Slice,
    is_subset,
    set_from_geometry,
)

x = Variable("x", dim=2)
t = Variable("t", dim=1)
Omega = Domain("Omega", ambient=x)
dO = Boundary(Omega)
Gamma_D = Part("Gamma_D", dO)
Gamma_N = Part("Gamma_N", dO)
Interval = Domain("I", ambient=t)
unit_square = Rectangle(x, origin=(0.0, 0.0), width=1.0, height=1.0)
unit_time = IntervalGeometry(t, 0.0, 1.0)


class TestCoordinates:
    def test_accepts_variable_with_positive_int_dim(self):
        assert check_coordinate(x) is x

    @pytest.mark.parametrize(
        "bad", [Variable("t"), Variable("z", dim=0), Variable("w", dim=(2, 2))]
    )
    def test_rejects_variables_that_are_not_coordinates(self, bad):
        with pytest.raises(ValueError):
            check_coordinate(bad)

    def test_rejects_non_variables(self):
        with pytest.raises(ValueError):
            check_coordinate("x")


class TestSetStructure:
    def test_domain_lives_in_its_ambient_coordinate(self):
        assert Omega.axes == frozenset({x})
        assert Omega.codim == 0

    def test_boundary_keeps_axes_and_adds_one_to_codim(self):
        assert dO.axes == frozenset({x})
        assert dO.codim == 1

    def test_part_takes_axes_and_codim_of_its_set(self):
        assert Gamma_D.axes == dO.axes
        assert Gamma_D.codim == 1

    def test_product_unions_axes_and_adds_codim(self):
        space_time = Product((Omega, Interval))
        assert space_time.axes == frozenset({x, t})
        assert space_time.codim == 0

    def test_slice_drops_one_dimension(self):
        initial = Product((Omega, Slice(t, 0.0)))
        assert initial.axes == frozenset({x, t})
        assert initial.codim == 1

    def test_product_rejects_shared_coordinates(self):
        with pytest.raises(ValueError):
            Product((Omega, Omega))

    def test_product_rejects_no_factors(self):
        with pytest.raises(ValueError):
            Product(())

    def test_slice_needs_a_scalar_coordinate(self):
        with pytest.raises(ValueError):
            Slice(x, 0.0)

    def test_domain_needs_a_coordinate(self):
        with pytest.raises(ValueError):
            Domain("bad", ambient=Variable("q"))


class TestEquality:
    def test_equal_structure_is_equal(self):
        assert Boundary(Domain("Omega", ambient=x)) == dO

    def test_coordinates_compare_by_name_and_dim(self):
        assert Domain("Omega", ambient=Variable("x", dim=2)) == Omega

    def test_sets_are_hashable(self):
        assert len({Omega, Domain("Omega", ambient=x), dO}) == 2


class TestIsSubset:
    def test_a_set_is_a_subset_of_itself(self):
        assert is_subset(dO, dO)

    def test_part_is_a_subset_of_the_boundary_it_belongs_to(self):
        assert is_subset(Gamma_D, dO)

    def test_nested_part_is_a_subset_of_the_outer_set(self):
        assert is_subset(Part("inner", Gamma_D), dO)

    def test_boundary_is_not_inside_the_open_domain(self):
        assert not is_subset(dO, Omega)
        assert not is_subset(Gamma_D, Omega)

    def test_different_parts_are_not_subsets_of_each_other(self):
        assert not is_subset(Gamma_D, Gamma_N)

    def test_slice_lies_in_its_coordinate_domain(self):
        assert is_subset(Slice(t, 0.0), Interval)
        assert not is_subset(Slice(t, 0.0), Omega)

    def test_product_is_a_subset_factorwise_in_any_order(self):
        assert is_subset(Product((Interval, Omega)), Product((Omega, Interval)))

    def test_initial_slice_is_a_subset_of_space_time(self):
        initial = Product((Omega, Slice(t, 0.0)))
        space_time = Product((Omega, Interval))
        assert is_subset(initial, space_time)
        assert not is_subset(space_time, initial)


class TestDomainGeometry:
    def test_coordinate_comes_from_the_geometry(self):
        domain = Domain("Omega", geometry=unit_square)
        assert domain.ambient == x
        assert domain.axes == frozenset({x})

    def test_geometry_is_not_part_of_equality_or_hash(self):
        other = Rectangle(Variable("x", dim=2), origin=(0.0, 0.0), width=2.0, height=1.0)
        assert Domain("Omega", geometry=unit_square) == Domain("Omega", geometry=other)
        assert Domain("Omega", geometry=unit_square) == Omega
        assert hash(Domain("Omega", geometry=unit_square)) == hash(Omega)

    def test_ambient_must_match_the_geometry(self):
        with pytest.raises(ValueError):
            Domain("Omega", ambient=Variable("y", dim=2), geometry=unit_square)

    def test_domain_needs_an_ambient_or_a_geometry(self):
        with pytest.raises(ValueError):
            Domain("Omega")

    def test_only_continuous_geometries_are_accepted(self):
        with pytest.raises(TypeError):
            Domain("Omega", geometry=Geometry(variable=x))


class TestSetFromGeometry:
    def test_continuous_geometry_becomes_a_named_domain(self):
        domain = set_from_geometry(unit_square, "Omega")
        assert domain == Omega
        assert domain.geometry is unit_square

    def test_product_geometry_becomes_a_product_of_named_factors(self):
        space_time = set_from_geometry(unit_square * unit_time)
        assert space_time == Product((Domain("x", ambient=x), Domain("t", ambient=t)))
        assert space_time.axes == frozenset({x, t})

    def test_product_rejects_a_name(self):
        with pytest.raises(ValueError):
            set_from_geometry(unit_square * unit_time, "Omega")

    def test_continuous_geometry_needs_a_name(self):
        with pytest.raises(ValueError):
            set_from_geometry(unit_square)

    def test_non_continuous_factor_is_rejected(self):
        with pytest.raises(TypeError):
            set_from_geometry(unit_square * Geometry(variable=t))
