import pytest

from pioneer.config.variables import Variable

from pioneer.config.configuration_base import (
    Axes,
    DataConfigMismatchError,
    DataConfigDtypeMismatchError,
    DataConfiguration,
    DTypeUnit,
    BatchAxes,
    FeatureAxes,
)


class TestAxesShapeMatching:

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


class TestConfigUnify:

    def feature_dummy(self):
        return FeatureAxes(Variable("x", 1))

    def test_config_create(self):
        test_type = DTypeUnit(None, [self.feature_dummy()])
        config_1 = DataConfiguration(test_type)
        assert isinstance(config_1.dtype_units, list)
        assert config_1.dtype_units[0] == test_type

    def test_unify_same_config(self):
        config_1 = DataConfiguration(
            DTypeUnit(None, [BatchAxes((10, 5)), self.feature_dummy()])
        )
        new_config = config_1.unify(config_1)
        assert len(new_config.dtype_units) == 1
        assert new_config.dtype_units[0].dtype is None
        assert new_config.dtype_units[0].axes[0].shape == (10, 5)  # type: ignore

    def test_unify_same_config_multiple_axis(self):
        f_axis = FeatureAxes(Variable("x", 7))
        config_1 = DataConfiguration(
            DTypeUnit(list, [BatchAxes((10, 5)), BatchAxes((3,)), f_axis])
        )
        new_config = config_1.unify(config_1)
        assert len(new_config.dtype_units) == 1
        assert new_config.dtype_units[0].dtype == list
        assert len(new_config.dtype_units[0].axes) == 2
        assert new_config.dtype_units[0].axes[0].shape == (10, 5, 3)  # type: ignore
        assert isinstance(new_config.dtype_units[0].axes[1], FeatureAxes)
        assert new_config.dtype_units[0].axes[1].shape == (7,)  # type: ignore

    def test_unify_same_config_multiple_dtypes(self):
        f_axis = FeatureAxes(Variable("x", 7))
        config_1 = DataConfiguration(
            [
                DTypeUnit(list, [BatchAxes((10, 5)), f_axis]),
                DTypeUnit(None, [BatchAxes((50, 50, 50))]),
            ]
        )
        new_config = config_1.unify(config_1)
        assert len(new_config.dtype_units) == 2
        assert new_config.dtype_units[0].dtype == list
        assert new_config.dtype_units[1].dtype is None
        assert len(new_config.dtype_units[0].axes) == 2
        assert new_config.dtype_units[0].axes[0].shape == (10, 5)  # type: ignore
        assert new_config.dtype_units[0].axes[1].shape == (7,)  # type: ignore
        assert len(new_config.dtype_units[1].axes) == 1
        assert new_config.dtype_units[1].axes[0].shape == (50, 50, 50)  # type: ignore

    def test_unify_wrong_dtypes(self):
        config_1 = DataConfiguration(
            DTypeUnit(None, [BatchAxes((10, 5)), self.feature_dummy()])
        )
        config_2 = DataConfiguration(
            DTypeUnit(list, [BatchAxes((10, 5)), self.feature_dummy()])
        )
        with pytest.raises(DataConfigDtypeMismatchError):
            _ = config_1.unify(config_2)

    def test_unify_wrong_shapes_in_axis(self):
        config_1 = DataConfiguration(
            DTypeUnit(None, [BatchAxes((10, 5)), self.feature_dummy()])
        )
        config_2 = DataConfiguration(
            DTypeUnit(None, [BatchAxes((4, 5)), self.feature_dummy()])
        )
        with pytest.raises(DataConfigMismatchError):
            _ = config_1.unify(config_2)

    def test_unify_wrong_axis(self):
        config_1 = DataConfiguration(
            DTypeUnit(None, [BatchAxes((7,)), self.feature_dummy()])
        )
        config_2 = DataConfiguration(DTypeUnit(None, [self.feature_dummy()]))
        with pytest.raises(DataConfigMismatchError):
            _ = config_1.unify(config_2)

    def test_unify_configs(self):
        config_1 = DataConfiguration(
            DTypeUnit(None, [BatchAxes((10, 5)), self.feature_dummy()])
        )
        config_2 = DataConfiguration(
            DTypeUnit(None, [BatchAxes((..., 5)), self.feature_dummy()])
        )
        new_config = config_1.unify(config_2)
        assert len(new_config.dtype_units) == 1
        assert new_config.dtype_units[0].dtype is None
        assert new_config.dtype_units[0].axes[0].shape == (10, 5)  # type: ignore

    def test_unify_configs_three_axis(self):
        config_1 = DataConfiguration(
            DTypeUnit(
                list, [BatchAxes((10, 5)), BatchAxes((1, ..., 7)), self.feature_dummy()]
            )
        )
        config_2 = DataConfiguration(
            DTypeUnit(
                list, [BatchAxes((..., 5)), BatchAxes((1, 1, 7)), self.feature_dummy()]
            )
        )
        new_config = config_1.unify(config_2)
        assert len(new_config.dtype_units) == 1
        assert new_config.dtype_units[0].dtype == list
        assert new_config.dtype_units[0].axes[0].shape == (10, 5, 1, 1, 7)  # type: ignore

    def test_unify_configs_three_axis_not_collapsable(self):
        config_1 = DataConfiguration(
            DTypeUnit(
                list,
                [BatchAxes((10, 5, ...)), BatchAxes((1, ..., 7)), self.feature_dummy()],
            )
        )
        config_2 = DataConfiguration(
            DTypeUnit(
                list,
                [BatchAxes((..., 5)), BatchAxes((..., 1, 1, 7)), self.feature_dummy()],
            )
        )
        new_config = config_1.unify(config_2)
        assert len(new_config.dtype_units) == 1
        assert new_config.dtype_units[0].dtype == list
        assert new_config.dtype_units[0].axes[0].shape == (10, 5)  # type: ignore
        assert new_config.dtype_units[0].axes[1].shape == (1, 1, 7)  # type: ignore

    # TODO: This test should also work but it contradicts a bit the previous one...
    # def test_unify_configs_three_axis_not_collapsable_2(self):
    #     config_1 = DataConfiguration(
    #         DTypeUnit(
    #             list,
    #             [BatchAxes((10, 5, ...)), BatchAxes((1, ..., 7)), self.feature_dummy()],
    #         )
    #     )
    #     config_2 = DataConfiguration(
    #         DTypeUnit(
    #             list, [BatchAxes((..., 5)), BatchAxes((1, 1, 7)), self.feature_dummy()]
    #         )
    #     )
    #     new_config = config_1.unify(config_2)
    #     assert len(new_config.dtype_units) == 1
    #     assert new_config.dtype_units[0].dtype == list
    #     assert new_config.dtype_units[0].axes[0].shape == (10, 5)  # type: ignore
    #     assert new_config.dtype_units[0].axes[1].shape == (1, 1, 7)  # type: ignore


ttt = TestConfigUnify()
ttt.test_unify_same_config()
