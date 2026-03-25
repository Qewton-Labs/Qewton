import pytest

from pioneer.config.configuration_base import Axes, DataConfigMismatchError


class TestConfigShapeMatching:

    def test_unify_shapes_concrete(self):
        a = (2, 3, 4)
        b = (2, 3, 4)
        expected = (2, 3, 4)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_broadcasting(self):
        a = (2, 3, 4)
        b = (1, 3, 1)
        expected = (2, 3, 4)
        result = Axes.unify_shapes(a, b, broadcast_singleton=True)
        assert result == expected

    def test_unify_shapes_broadcasting_set_to_false(self):
        a = (2, 3, 4)
        b = (1, 3, 1)
        with pytest.raises(DataConfigMismatchError):
            Axes.unify_shapes(a, b, broadcast_singleton=False)

    def test_unify_shapes_broadcasting_not_working(self):
        a = (2, 3, 4)
        b = (2, 1, 5)
        with pytest.raises(DataConfigMismatchError):
            Axes.unify_shapes(a, b, broadcast_singleton=True)

    def test_unify_shapes_with_none(self):
        a = (2, None, 4)
        b = (2, 3, None)
        expected = (2, 3, 4)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis(self):
        a = (2, ..., 4)
        b = (2, 3, 4)
        expected = (2, 3, 4)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis_broadcasting(self):
        a = (2, ..., 4)
        b = (1, 3, 1)
        expected = (2, 3, 4)
        result = Axes.unify_shapes(a, b, broadcast_singleton=True)
        assert result == expected

    def test_unify_shapes_with_ellipsis_at_start(self):
        a = (..., 4)
        b = (2, 7, 4)
        expected = (2, 7, 4)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis_not_matching(self):
        a = (..., 5)
        b = (2, 7, 4)
        with pytest.raises(DataConfigMismatchError):
            Axes.unify_shapes(a, b)

    def test_unify_shapes_with_ellipsis_at_end(self):
        a = (2, ...)
        b = (2, 7, 4)
        expected = (2, 7, 4)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis_only(self):
        a = (...,)
        b = (12, 7, 48)
        expected = (12, 7, 48)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis_in_both_mixed(self):
        a = (2, ..., 4)
        b = (..., 4)
        expected = (2, ..., 4)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis_in_both_mixed_2(self):
        a = (..., 10, 4)
        b = (4, ...)
        expected = (4, 10, 4)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis_in_both_mixed_3(self):
        a = (4, 10, ...)
        b = (..., 4)
        expected = (4, 10, 4)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis_in_both_at_start(self):
        a = (..., 4)
        b = (..., 4)
        expected = (..., 4)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis_in_both_at_start_2(self):
        a = (..., 5, 4)
        b = (..., 4)
        expected = (..., 5, 4)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis_in_both_at_end(self):
        a = (2, 3, ...)
        b = (2, 3, ...)
        expected = (2, 3, ...)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis_in_both_at_end_2(self):
        a = (2, 30, ...)
        b = (2, ...)
        expected = (2, 30, ...)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_with_ellipsis_at_start_and_end(self):
        a = (..., 4)
        b = (5, 4, ...)
        expected = (5, 4)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_high_dim_with_broadcasting(self):
        a = (2, 3, 4, 5, 6)
        b = (1, 1, 4, 1, 6)
        expected = (2, 3, 4, 5, 6)
        result = Axes.unify_shapes(a, b, broadcast_singleton=True)
        assert result == expected

    def test_unify_shapes_complex_none_ellipsis_mix(self):
        a = (2, None, ..., 5)
        b = (2, 3, 4, 5)
        expected = (2, 3, 4, 5)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_ellipsis_with_none_broadcasting(self):
        a = (2, None, ..., 6)
        b = (1, 3, 4, 5, 1)
        expected = (2, 3, 4, 5, 6)
        result = Axes.unify_shapes(a, b, broadcast_singleton=True)
        assert result == expected

    def test_unify_shapes_concrete_without_ellipsis(self):
        a = (2, 3, ..., 5, 6)
        b = (2, 3, 5, 6)
        expected = (2, 3, 5, 6)
        result = Axes.unify_shapes(a, b)
        assert result == expected

    def test_unify_shapes_center_ellipsis_removed(self):
        a = (2, 3, ..., 5, 6)
        b = (..., 3, 5, 6)
        expected = (2, 3, 5, 6)
        result = Axes.unify_shapes(a, b)
        assert result == expected
