import pytest
import torch

from qewton.config.variables import Variable
from qewton.geometries.discrete.index_grid_geometry import IndexGridGeometry


def _point_filter(shape, exclude=()):
    """point_filter must already be a backend tensor (TensorType), same
    contract as GridGeometry's own point_filter - not a raw numpy array."""
    pf = torch.ones(shape + (1,), dtype=torch.bool)
    for idx in exclude:
        pf[idx + (0,)] = False
    return pf


class TestConstruction:
    def test_dim_mismatch_raises(self):
        with pytest.raises(AssertionError, match="must match"):
            IndexGridGeometry(Variable("g", 3), (3, 4))

    def test_is_not_materialized_before_first_access(self):
        geo = IndexGridGeometry(Variable("g", 2), (3, 4))
        assert geo.is_materialized is False

    def test_default_point_filter_includes_every_point(self):
        geo = IndexGridGeometry(Variable("g", 2), (3, 4))
        assert bool(geo.point_filter.all())
        assert tuple(geo.point_filter.shape) == (3, 4, 1)

    def test_explicit_point_filter_is_respected(self):
        pf = _point_filter((3, 4), exclude=[(0, 0)])
        geo = IndexGridGeometry(Variable("g", 2), (3, 4), point_filter=pf)
        assert not bool(geo.point_filter[0, 0, 0])
        assert bool(geo.point_filter[1, 0, 0])

    def test_mismatched_point_filter_shape_raises(self):
        with pytest.raises(AssertionError, match="Filter and grid shape"):
            IndexGridGeometry(
                Variable("g", 2), (3, 4), point_filter=_point_filter((2, 4))
            )


class TestLazyDiscretizationPoints:
    def test_materializes_on_first_access_and_caches(self):
        geo = IndexGridGeometry(Variable("g", 2), (3, 4))
        points = geo.discretization_points
        assert geo.is_materialized is True
        assert points is geo.discretization_points  # same cached object

    def test_points_are_the_grids_own_indices(self):
        geo = IndexGridGeometry(Variable("g", 2), (3, 4))
        points = geo.discretization_points
        assert tuple(points.shape) == (3, 4, 2)
        assert list(points[2, 3]) == [2, 3]
        assert list(points[0, 0]) == [0, 0]

    def test_points_are_integer_dtype(self):
        geo = IndexGridGeometry(Variable("g", 2), (3, 4))
        assert "int" in str(geo.discretization_points.dtype)

    def test_setter_overrides_the_lazy_value(self):
        """DiscreteGeometry.__init__ and _move_data both assign through this
        setter - it must land in the lazy slot, not shadow the property."""
        geo = IndexGridGeometry(Variable("g", 2), (3, 4))
        geo.discretization_points = "sentinel"
        assert geo.discretization_points == "sentinel"
        assert geo.is_materialized is True


class TestDeviceMove:
    def test_moving_before_materialization_stays_lazy(self):
        geo = IndexGridGeometry(Variable("g", 2), (3, 4))
        geo._move_data("cpu")
        assert geo.is_materialized is False
        points = geo.discretization_points
        assert tuple(points.shape) == (3, 4, 2)

    def test_moving_after_materialization_moves_the_existing_points(self):
        geo = IndexGridGeometry(Variable("g", 2), (3, 4))
        points_before = geo.discretization_points
        geo._move_data("cpu")
        assert tuple(geo.discretization_points.shape) == tuple(points_before.shape)


class TestBoundingBoxAndVolume:
    def test_bounding_box_spans_zero_to_extent_minus_one(self):
        geo = IndexGridGeometry(Variable("g", 2), (3, 4))
        lo, hi = geo.bounding_box()
        assert list(lo) == [0.0, 0.0]
        assert list(hi) == [2.0, 3.0]

    def test_volume_is_the_unit_cell_count_when_fully_included(self):
        geo = IndexGridGeometry(Variable("g", 2), (3, 4))
        # (3-1) * (4-1) = 6 unit cells
        assert float(geo.volume()) == 6.0

    def test_volume_excludes_cells_touching_a_filtered_out_point(self):
        pf = _point_filter((3, 4), exclude=[(0, 0)])
        geo = IndexGridGeometry(Variable("g", 2), (3, 4), point_filter=pf)
        assert float(geo.volume()) == 5.0

    def test_volume_does_not_require_materializing_points(self):
        """Cell volumes for an index grid follow purely from point_filter -
        no coordinate array is needed, so computing volume() must not force
        materialization."""
        geo = IndexGridGeometry(Variable("g", 2), (3, 4))
        geo.volume()
        assert geo.is_materialized is False
