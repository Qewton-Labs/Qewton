from __future__ import annotations

from .axes import Axes, EllipsisAxes, _match_remainder
from .errors import DataConfigMismatchError


class DataConfiguration:
    def __init__(self, *axes: Axes | EllipsisAxes, dtype=None):
        self.axes = axes
        self.dtype = dtype

    def __str__(self):
        return f"DataConfig([{', '.join(str(a) for a in self.axes)}])"

    def unify_with(self, other: DataConfiguration) -> tuple[dict, dict]:
        print(self.axes, other.axes)
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
            print(a1, a2)
            try:
                if isinstance(a1, EllipsisAxes) or isinstance(a2, EllipsisAxes):
                    # We can not unify ellipsis directly, since they may need to
                    # consume other axis as well.
                    break
                unified_axes = Axes.unify_with(a1, a2)
                matching_axes_1[a1], matching_axes_2[a2] = unified_axes
            except DataConfigMismatchError as e:
                raise DataConfigMismatchError(
                    f"Axes {axes1} and {axes2} do not match and can not be unified.\
                        Mismatch at axes {a1} and {a2}."
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
                Axes, list(remaining_middle1), list(remaining_middle2)
            )
        else:
            middle2, middle1 = _match_remainder(
                Axes, list(remaining_middle2), list(remaining_middle1)
            )

        return middle1, middle2

    def has_changed(self, new_config_dict: dict) -> bool:
        # TODO
        return True
