import pytest

from pioneer.config.axis import (
    Axes,
    EllipsisDim,
    AxesDim,
    BatchAxes,
    FeatureAxes,
    GeometryAxes,
)
from pioneer.config.data_configurations import DataConfiguration
from pioneer.config.errors import DataConfigMismatchError


class TestAxes:

    def test_unify_axes_exact_match(self):
        a = Axes((2, 3))
        b = Axes((2, 3))

        unified = a.unify_with(b)

        assert isinstance(unified, Axes)
        assert [dim.size for dim in unified.shape] == [2, 3]

    def test_unify_axes_with_broadcastable_dims(self):
        a = Axes((AxesDim(1), 3))
        b = Axes((AxesDim(2), 3))

        unified = a.unify_with(b)

        assert [dim.size for dim in unified.shape] == [2, 3]
        assert all(isinstance(dim, AxesDim) for dim in unified.shape)

    def test_unify_axes_with_ellipsis(self):
        a = Axes((1, EllipsisDim(), 3))
        b = Axes((1, 2, 3))

        unified = a.unify_with(b)

        assert [dim.size for dim in unified.shape] == [1, 2, 3]
        assert isinstance(unified.shape[1], AxesDim)

    def test_unify_axes_type_mismatch_raises(self):
        a = Axes((2, 3))
        b = BatchAxes((2, 3))

        with pytest.raises(DataConfigMismatchError):
            a.unify_with(b)

    def test_unify_batch_axes_same_shape(self):
        a = BatchAxes((1, 10))
        b = BatchAxes((AxesDim(1), 10))

        unified = a.unify_with(b)

        assert isinstance(unified, BatchAxes)
        assert [dim.size for dim in unified.shape] == [1, 10]

    def test_unify_batch_axes_conflicting_shape_raises(self):
        a = BatchAxes((2, 4))
        b = BatchAxes((3, 4))

        with pytest.raises(DataConfigMismatchError):
            a.unify_with(b)

    def test_unify_feature_axes(self):
        a = FeatureAxes(shape=(AxesDim(1), 5))
        b = FeatureAxes(shape=(2, 5))

        unified = a.unify_with(b)

        assert isinstance(unified, FeatureAxes)
        assert [dim.size for dim in unified.shape] == [2, 5]

    def test_unify_geometry_axes(self):
        a = GeometryAxes(shape=(2, 3))
        b = GeometryAxes(shape=(AxesDim(2), 3))

        unified = a.unify_with(b)

        assert isinstance(unified, GeometryAxes)
        assert [dim.size for dim in unified.shape] == [2, 3]

    def test_geometry_axes_type_mismatch_raises(self):
        a = GeometryAxes(shape=(2, 3))
        b = Axes((2, 3))

        with pytest.raises(DataConfigMismatchError):
            a.unify_with(b)

    def test_axes_dim_unify_conflict_raises(self):
        dim_a = AxesDim(2)
        dim_b = AxesDim(3)

        with pytest.raises(DataConfigMismatchError):
            dim_a.unify_with(dim_b)


class TestDataConfiguration:

    def test_unify_exact_match(self):
        a = DataConfiguration(Axes((1, 2, 3)), dtype=float)
        b = DataConfiguration(Axes((1, 2, 3)), dtype=float)

        unified = a.unify_with(b)

        assert isinstance(unified, DataConfiguration)
        assert unified.dtype == float
        assert isinstance(unified.axes[0], Axes)
        assert tuple(dim.size for dim in unified.axes[0].shape) == (1, 2, 3)

    def test_unify_dtype_mismatch_raises(self):
        a = DataConfiguration(Axes((1, 2)), dtype=float)
        b = DataConfiguration(Axes((1, 2)), dtype=int)

        with pytest.raises(DataConfigMismatchError):
            a.unify_with(b)

    def test_unify_middle_ellipsis_matches_extra_axes(self):
        a = DataConfiguration(
            BatchAxes((1, 10)),
            EllipsisDim(),
            FeatureAxes(shape=(2, 5)),
            dtype=float,
        )
        b = DataConfiguration(
            BatchAxes((1, 10)),
            Axes((3, 4)),
            FeatureAxes(shape=(2, 5)),
            dtype=float,
        )

        unified = a.unify_with(b)

        assert isinstance(unified.axes[0], BatchAxes)
        assert isinstance(unified.axes[1], Axes)
        assert isinstance(unified.axes[2], FeatureAxes)
        assert tuple(dim.size for dim in unified.axes[1].shape) == (3, 4)

    def test_unify_with_only_ellipsis_wildcard(self):
        a = DataConfiguration(
            BatchAxes((1, 10)),
            EllipsisDim(),
            FeatureAxes(shape=(2, 5)),
            dtype=float,
        )
        b = DataConfiguration(EllipsisDim(), dtype=float)

        unified = a.unify_with(b)

        assert len(unified.axes) == 3
        assert isinstance(unified.axes[0], BatchAxes)
        assert isinstance(unified.axes[1], EllipsisDim)
        assert isinstance(unified.axes[2], FeatureAxes)

    def test_unify_with_ellipsis_prefix_feature_suffix(self):
        a = DataConfiguration(
            EllipsisDim(),
            FeatureAxes(shape=(2, 5)),
            dtype=float,
        )
        b = DataConfiguration(
            Axes((1, 2)),
            Axes((3, 4)),
            FeatureAxes(shape=(2, EllipsisDim())),
            dtype=float,
        )

        unified = a.unify_with(b)

        assert len(unified.axes) == 3
        assert not isinstance(unified.axes[0], EllipsisDim)
        assert tuple(dim.size for dim in unified.axes[0].shape) == (1, 2)
        assert not isinstance(unified.axes[1], EllipsisDim)
        assert tuple(dim.size for dim in unified.axes[1].shape) == (3, 4)
        assert isinstance(unified.axes[2], FeatureAxes)
        assert tuple(dim.size for dim in unified.axes[2].shape) == (2, 5)

    def test_unify_with_both_sides_ellipsis_preserves_wildcard(self):
        a = DataConfiguration(
            BatchAxes((1, 10)),
            EllipsisDim(),
            FeatureAxes(shape=(2, 5)),
            dtype=float,
        )
        b = DataConfiguration(
            BatchAxes((1, 10)),
            EllipsisDim(),
            FeatureAxes(shape=(2, 5)),
            dtype=float,
        )

        unified = a.unify_with(b)

        assert len(unified.axes) == 3
        assert isinstance(unified.axes[1], EllipsisDim)

    def test_unify_fails_without_ellipsis_for_extra_axis(self):
        a = DataConfiguration(
            BatchAxes((1, 10)),
            FeatureAxes(shape=(2, 5)),
            dtype=float,
        )
        b = DataConfiguration(
            BatchAxes((1, 10)),
            Axes((3, 4)),
            FeatureAxes(shape=(2, 5)),
            dtype=float,
        )

        with pytest.raises(DataConfigMismatchError):
            a.unify_with(b)

    def test_unify_with_ellipsis_at_end(self):
        a = DataConfiguration(Axes((1, 2, EllipsisDim())), dtype=float)
        b = DataConfiguration(Axes((1, 2, 3, 4)), dtype=float)

        unified = a.unify_with(b)

        assert isinstance(unified.axes[0], Axes)
        assert tuple(dim.size for dim in unified.axes[0].shape) == (1, 2, 3, 4)

    def test_unify_axis_type_mismatch_raises(self):
        a = DataConfiguration(Axes((1, 2)), dtype=float)
        b = DataConfiguration(BatchAxes((1, 2)), dtype=float)

        with pytest.raises(DataConfigMismatchError):
            a.unify_with(b)
