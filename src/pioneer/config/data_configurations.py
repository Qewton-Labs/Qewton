from __future__ import annotations

from .axes import Axes, EllipsisAxes, EllipsisDim, _match_remainder, FeatureAxes
from .errors import DataConfigMismatchError


class DataConfiguration:
    def __init__(self, *axes: Axes | EllipsisAxes, dtype=None):
        self.axes = axes
        self.dtype = dtype
        assert (
            len([f for f in axes if isinstance(f, FeatureAxes)]) <= 1
        ), "A DataConfig can have at most one FeatureAxes."

    @classmethod
    def empty(cls):
        return DataConfiguration(EllipsisAxes())

    def __str__(self):
        return f"DataConfig([{', '.join(str(a) for a in self.axes)}])"

    @property
    def feature_axes(self):
        for axes in self.axes:
            if isinstance(axes, FeatureAxes):
                return axes
        return None

    def unify_with(self, other: DataConfiguration) -> tuple[dict, dict]:
        if self.dtype != other.dtype:
            raise DataConfigMismatchError(
                f"Found different data types {self.dtype} and {other.dtype}."
            )

        # First we check if they match from the end
        matching_end_self, matching_end_other = self._match_axes(
            reversed(self.axes), reversed(other.axes)
        )
        if len(matching_end_self) == len(self.axes) and len(matching_end_other) == len(
            other.axes
        ):
            #  fully matched
            return matching_end_self, matching_end_other
        matching_start_self, matching_start_other = self._match_axes(
            self.axes, other.axes
        )
        # Lastly we check the remaining middle part for compatibility (it has
        # to contain an ellipsis in at least one of the shapes to be compatible)
        remaining_middle_self = self.axes[
            len(matching_start_self) : len(self.axes) - len(matching_end_self)
        ]
        remaining_middle_other = other.axes[
            len(matching_start_other) : len(other.axes) - len(matching_end_other)
        ]
        if not any(
            isinstance(axes, EllipsisAxes) for axes in remaining_middle_self
        ) and not any(isinstance(axes, EllipsisAxes) for axes in remaining_middle_other):
            raise DataConfigMismatchError(
                f"Axes {self.axes} and {other.axes} do not match and can not be unified."
            )
        matching_middle_self, matching_middle_other = self._match_middle_axes(
            remaining_middle_self, remaining_middle_other
        )
        return (
            matching_start_self | matching_middle_self | matching_end_self,
            matching_start_other | matching_middle_other | matching_end_other,
        )

    @classmethod
    def _match_axes(cls, axes1, axes2) -> tuple[dict, dict]:
        matching_axes_1 = {}
        matching_axes_2 = {}
        for a1, a2 in zip(axes1, axes2):
            try:
                if isinstance(a1, EllipsisAxes) or isinstance(a2, EllipsisAxes):
                    # We can not unify ellipsis directly, since they may need to
                    # consume other axis as well.
                    break
                unified_axes = Axes.unify_with(a1, a2)
                matching_axes_1[a1], matching_axes_2[a2] = unified_axes
            except DataConfigMismatchError as e:
                axes_1_str = f"[{', '.join(str(a) for a in axes1)}]"
                axes_2_str = f"[{', '.join(str(a) for a in axes2)}]"
                raise DataConfigMismatchError(
                    f"Axes {axes_1_str} and {axes_2_str} do not match and can not be "
                    + f"unified. Mismatch at axes {a1} and {a2}."
                ) from e
        return matching_axes_1, matching_axes_2

    @classmethod
    def _match_middle_axes(
        cls, remaining_middle1, remaining_middle2
    ) -> tuple[dict, dict]:
        # If one shape only is an ellipsis we are done:
        if len(remaining_middle1) == 1 and isinstance(remaining_middle1[0], EllipsisAxes):
            return {remaining_middle1[0]: remaining_middle2}, {
                k: k for k in remaining_middle2
            }
        if len(remaining_middle2) == 1 and isinstance(remaining_middle2[0], EllipsisAxes):
            return {k: k for k in remaining_middle1}, {
                remaining_middle2[0]: remaining_middle1
            }

        # Determine which side has ellipsis at start vs end
        if isinstance(remaining_middle1[0], EllipsisAxes):
            middle1, middle2 = _match_remainder(
                Axes, list(remaining_middle1), list(remaining_middle2), EllipsisAxes
            )
        else:
            middle2, middle1 = _match_remainder(
                Axes, list(remaining_middle2), list(remaining_middle1), EllipsisAxes
            )

        return middle1, middle2

    def update_config(
        self, new_config_dict: dict[Axes, Axes | dict | tuple[Axes, ...]]
    ) -> bool:
        """Updates the current config inplace. Returns whether anything has changed.

        Parameters:
            new_config_dict (dict[Axes, Axes | dict | tuple[Axes, ...]): A dictionary
                containing for each original axis how it should be changed/updated.

        Returns:
            bool: If the configuration has changed in this update, or stayed the same.
        """
        changed_config = False
        new_axes_list = []
        for axes in self.axes:
            # Check if axes is even updated
            if not axes in new_config_dict:
                new_axes_list.append(axes)
                continue

            new_axes = new_config_dict[axes]
            if new_axes == axes:
                # If the element is the same we don't have to do anything
                new_axes_list.append(axes)
            elif isinstance(new_axes, dict):
                # If the new config is a dict, the general axis is the same
                # and we have to update the inner axis dimensions
                new_axes_list.append(axes)
                updated_axes = axes.update_axes(new_axes)
                changed_config |= updated_axes
            elif isinstance(axes, EllipsisAxes):
                # Then the current axes is only an EllipsisAxes and we can
                # replace it with the new axes
                if isinstance(new_axes, tuple | list):
                    new_axes_list.extend(list(new_axes))
                    changed_config = True
                    # Maybe its just tuple containing an EllipsisAxes
                    if len(new_axes) == 1:
                        changed_config = not isinstance(new_axes[0], EllipsisAxes)
                else:
                    new_axes_list.append(new_axes)
                    changed_config = not isinstance(new_axes, EllipsisAxes)

        self.axes = tuple(new_axes_list)
        return changed_config

    def get_variable_slice(self, variable):
        slc: list = []
        feature_slice = None
        for axes in self.axes:
            if isinstance(axes, FeatureAxes):
                feature_slice = axes.variables.get_slice(variable)
                if isinstance(feature_slice, tuple):
                    slc.extend(feature_slice)
                else:
                    slc.append(feature_slice)
            elif isinstance(axes, EllipsisAxes) or any(
                isinstance(d, EllipsisDim) for d in axes.shape
            ):
                if feature_slice is None:
                    slc = [...]
                else:
                    assert (
                        Ellipsis not in slc
                    ), "Can not uniquely find the feature axes location. \
                        Too many ellipses."
                    slc.append(...)
                    break
            else:
                slc.extend([slice(None)] * len(axes.shape))  # type: ignore
        return tuple(slc)
