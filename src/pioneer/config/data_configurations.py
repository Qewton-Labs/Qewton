from __future__ import annotations

from .axes import Axes, EllipsisAxes
from .errors import DataConfigMismatchError


class DataConfiguration:
    def __init__(self, *axes: Axes | EllipsisAxes, dtype=None):
        self.axes = axes
        self.dtype = dtype

    def __str__(self):
        return f"DataConfig([{', '.join(str(a) for a in self.axes)}])"

    def unify_with(self, other: DataConfiguration) -> DataConfiguration:
        if self.dtype != other.dtype:
            raise DataConfigMismatchError(
                f"Found different data types {self.dtype} and {other.dtype}."
            )

        # First we check if they match from the end
        matching_end = self._match_axes(reversed(self.axes), reversed(other.axes))
        matching_end.reverse()
        if len(matching_end) == len(self.axes) and len(matching_end) == len(other.axes):
            #  fully matched
            return DataConfiguration(*matching_end, dtype=self.dtype)
        matching_start = self._match_axes(self.axes, other.axes)
        # Lastly we check the remaining middle part for compatibility (it has
        # to contain an ellipsis in at least one of the shapes to be compatible)
        remaining_middle1 = self.axes[
            len(matching_start) : len(self.axes) - len(matching_end)
        ]
        remaining_middle2 = other.axes[
            len(matching_start) : len(other.axes) - len(matching_end)
        ]
        if not any(
            isinstance(axes, EllipsisAxes) for axes in remaining_middle1
        ) and not any(isinstance(axes, EllipsisAxes) for axes in remaining_middle2):
            raise DataConfigMismatchError(
                f"Axes {self.axes} and {other.axes} do not match and can not be unified."
            )
        matching_middle = self._match_middle_axes(remaining_middle1, remaining_middle2)
        return DataConfiguration(
            *(matching_start + matching_middle + matching_end), dtype=self.dtype
        )

    @classmethod
    def _match_axes(cls, axes1, axes2) -> list:
        matching_axes = []
        for a1, a2 in zip(axes1, axes2):
            try:
                if isinstance(a1, EllipsisAxes) or isinstance(a2, EllipsisAxes):
                    # We can not unify ellipsis directly, since they may need to
                    # consume other axis as well.
                    break
                unified_axes = Axes.unify_with(a1, a2)
                matching_axes.append(unified_axes)
            except DataConfigMismatchError as e:
                raise DataConfigMismatchError(
                    f"Axes {axes1} and {axes2} do not match and can not be unified.\
                        Mismatch at axes {a1} and {a2}."
                ) from e
        return matching_axes

    @classmethod
    def _match_middle_axes(cls, remaining_middle1, remaining_middle2) -> list:
        # If one shape only is an ellipsis we are done:
        if len(remaining_middle1) == 1 and isinstance(remaining_middle1[0], EllipsisAxes):
            return list(remaining_middle2)
        if len(remaining_middle2) == 1 and isinstance(remaining_middle2[0], EllipsisAxes):
            return list(remaining_middle1)

        matching_middle = []
        # Determine which side has ellipsis at start vs end
        if isinstance(remaining_middle1[0], EllipsisAxes):
            start_part, end_part = list(remaining_middle1), list(remaining_middle2)
        else:
            start_part, end_part = list(remaining_middle2), list(remaining_middle1)

        start_idx = len(start_part) - 1
        end_idx = len(end_part) - 2  # skip ellipsis
        added_end_shape = False

        while start_idx > 0 and end_idx >= 0:
            current = start_part[start_idx]
            try:
                Axes.unify_with(current, end_part[end_idx])  # check if they match
                # If we have a match, they could be the same axis:
                # check if all neighbors to the left also match
                # If yes we can just add everything together
                offset = end_idx + 1 - start_idx
                if offset >= 0:
                    try:
                        insert_axes = [
                            Axes.unify_with(start_part[i], end_part[offset - 1 + i])
                            for i in range(1, start_idx + 1)
                        ]
                        matching_middle.reverse()
                        matching_middle = (
                            end_part[: end_idx + 1 - start_idx]
                            + insert_axes
                            + matching_middle
                        )
                        start_idx = 0
                        added_end_shape = True
                        break
                    except DataConfigMismatchError:
                        pass
            except DataConfigMismatchError:
                pass

            matching_middle.append(current)
            start_idx -= 1

        if not added_end_shape:
            matching_middle.reverse()
            matching_middle = end_part[:-1] + matching_middle

        return matching_middle


class DynamicDataConfiguration(DataConfiguration):
    def __init__(self, source_data_config):
        self.source_data_config = source_data_config
        axes = [DynamicAxes(a) for a in source_data_config.axes]
        super().__init__(*source_data_config.axes, dtype=dtype)
