from __future__ import annotations
from typing import Any

from qewton.config.axes import (
    Axes,
    EllipsisAxes,
    EllipsisDim,
    _match_remainder,
    FeatureAxes,
    AxesDim,
)
from qewton.config.errors import DataConfigMismatchError
from qewton.config.variables import Variable

## TODO: could we simplify the config passing to use common AxesDim
# objects nearly around the whole graph? this would allow for less
# passing operations and less objects.


class DataConfiguration:
    """A *DataConfiguration* describes the expected structure of the data,
    including the axes and their dimensions, as well as the data type.
    It is used to ensure that the data being passed through the graph
    matches the expected format for different methods/algorithms.

    Args:
        *axes (Axes | EllipsisAxes): The axes that describe the
            structure of the data.
        dtype (_type_, optional): The datatype used in this
            configuration. Defaults to None.
    """

    def __init__(self, *axes: Axes | EllipsisAxes, dtype=None):
        self.axes = axes
        self.dtype = dtype
        assert (
            len([f for f in axes if isinstance(f, FeatureAxes)]) <= 1
        ), "A DataConfig can have at most one FeatureAxes."

    @classmethod
    def empty(cls):
        """Builds an empty data configuration with only an ellipsis axis.
        This can be used as a placeholder when no specific data configuration
        is needed or when the data structure is completely flexible.

        Returns:
            DataConfiguration: The empty data configuration.
        """
        return DataConfiguration(EllipsisAxes())

    @property
    def variables(self):
        """Returns all variables defined in the feature axes of this
        data configuration.

        Returns:
            list[Variables]: The list of variables.
        """
        for axes in self.axes:
            if isinstance(axes, FeatureAxes):
                return axes.variables
        return []

    @property
    def variable_name(self):
        """Returns the name of the variable defined in the feature axes of
        this data configuration.
        Returns:
            str: The name of the variable, or an empty string
                if no variable is defined.
        """
        for axes in self.axes:
            if isinstance(axes, FeatureAxes):
                return axes.variables.name
        return ""

    def __str__(self):
        """Returns a readable representation of the configuration.

        Returns:
            str: A string representation of the data configuration.
        """
        return f"DataConfig([{', '.join(str(a) for a in self.axes)}])"

    def __repr__(self):
        return ", ".join(str(a) for a in self.axes)

    @property
    def feature_axes(self) -> FeatureAxes | None:
        """Returns the feature axes of this configuration.

        Returns:
            FeatureAxes | None: The feature axes if defined, otherwise None.
        """
        for axes in self.axes:
            if isinstance(axes, FeatureAxes):
                return axes
        return None

    def replace_feature_axes(self, new_feature_axes: FeatureAxes):
        """Replaces the feature axes in this configuration with new feature axes.
        This happens inplace.

        Args:
            new_feature_axes (FeatureAxes): The new feature axes to replace the
                existing ones.
        """
        new_axes = []
        for axes in self.axes:
            if isinstance(axes, FeatureAxes):
                new_axes.append(new_feature_axes)
            else:
                new_axes.append(axes)
        self.axes = tuple(new_axes)

    @property
    def feature_idx(self) -> int:
        """Returns the index of the feature axes in the configuration.
        If no feature axes is present, or if the feature axes can not be
        uniquely identified due to the presence of ellipsis, returns -1.

        Returns:
            int: The index of the feature axes, or -1 if it can not be uniquely
                identified.
        """
        counter = 0
        for axes in self.axes:
            if isinstance(axes, EllipsisAxes):
                # Can not get values via an index if ellipsis is present
                return -1
            for dim in axes.shape:
                if isinstance(dim, EllipsisDim):
                    # Can not get values via an index if ellipsis is present
                    return -1
                if isinstance(axes, FeatureAxes):
                    return counter
                counter += 1
        return -1

    def set_dtype(self, new_dtype):
        self.dtype = new_dtype

    def get_axes_and_dim(self, idx: int) -> tuple[Axes | None, AxesDim | None]:
        """Returns the axes object and corresponding axes dimension at a
        given shape index.

        Args:
            idx (int): The index of the shape dimension for which to retrieve
            the axes and dimension.

        Returns:
            tuple[Axes | None, AxesDim | None]: The axes object and corresponding
                axes dimension at the given index, or None if they can not be
                uniquely identified due to the presence of ellipsis.
        """
        if idx >= 0:
            counter = 0
            for axes in self.axes:
                if isinstance(axes, EllipsisAxes):
                    # Can not get values via an index if ellipsis is present
                    return None, None
                for dim in axes.shape:
                    if isinstance(dim, EllipsisDim):
                        # Can not get values via an index if ellipsis is present
                        return None, None
                    if counter == idx:
                        return axes, dim
                    counter += 1
        else:
            counter = -1
            for axes in reversed(self.axes):
                if isinstance(axes, EllipsisAxes):
                    return None, None
                for dim in reversed(axes.shape):
                    if isinstance(dim, EllipsisDim):
                        return None, None
                    if counter == idx:
                        return axes, dim
                    counter -= 1
        return None, None

    def remove_dim(self, axis: Axes, dim: AxesDim):
        """Removes the dimension from an axis. If the axis is empty
        afterwards, the axis will also be removed.

        Args:
            axis (Axes): The axis from which to remove the dimension.
            dim (AxesDim): The dimension to remove from the axis.
        """
        for axes in self.axes:
            if axes == axis:
                axes.remove_dim(dim)
                if axes.is_empty:
                    axes_list = list(self.axes)
                    axes_list.remove(axes)
                    self.axes = tuple(axes_list)
                return

    def matches(self, other: DataConfiguration) -> bool:
        """Check if both data configurations describe the same data.

        Args:
            other (DataConfiguration): Other data configuration to compare with.

        Returns:
            bool: Whether the data configurations match,
                meaning they describe the same data structure and type.
        """
        if len(self.axes) != len(other.axes):
            return False
        for a1, a2 in zip(self.axes, other.axes):
            if not a1.matches(a2):
                return False
        return True

    def unify_with(self, other: DataConfiguration) -> tuple[dict, dict]:
        """Combine two DataConfigurations, two exchange information between.
        Both configurations can make the other one more specific. Ellipsis,
        Axes and AxesDim can be concretized by this process.

        Args:
            other (DataConfiguration): The other data configuration to unify with.

        Raises:
            DataConfigMismatchError: If the data configurations can not be
                unified due to incompatible axes, data types, or dimensions.

        Returns:
            tuple[dict, dict]: Two dictionaries containing the axes and dimensions
                from both original configurations as keys, and the unified axes
                and dimensions as values.
        """
        # TODO: in future we should check whether one is a subtype of the other
        if self.dtype != other.dtype:
            if self.dtype is Any:
                self.dtype = other.dtype
            elif other.dtype is Any:
                other.dtype = self.dtype
            else:
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

        Args:
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

    def get_variable_slice(self, variable: Variable) -> tuple:
        """Computes the slice indices to obtain the variables along
        the current configuration.

        Args:
            variable (Variable): The variable for which to compute the
                slice indices.

        Returns:
            tuple: The slice indices to obtain the variable along the
                current configuration.
        """
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

    def get_axes_range(self, axes: Axes) -> int | tuple[int, int]:
        try:
            return self._find_axes_idx(self.axes, axes)
        except ValueError:
            try:
                reverse_axis_idx = self._find_axes_idx(self.axes[::-1], axes)
                if isinstance(reverse_axis_idx, int):
                    return -1 - reverse_axis_idx
                else:
                    return -1 - reverse_axis_idx[1], -1 - reverse_axis_idx[0]
            except ValueError as exc:
                raise ValueError(f"Axis {axes} not found in data config {self}.") from exc

    @classmethod
    def _find_axes_idx(cls, axes_list, searched_axes: Axes) -> int | tuple[int, int]:
        counter = 0
        for axes in axes_list:
            if axes is searched_axes:
                if len(axes.shape) == 1:
                    return counter
                else:
                    return (counter, counter + len(axes.shape))
            if isinstance(axes, EllipsisAxes) or any(
                isinstance(d, EllipsisDim) for d in axes.shape
            ):
                raise ValueError
            counter += len(axes.shape)
        raise ValueError

    def get_slice(self, data_config: DataConfiguration):
        try:
            axis_idx, slc = self._find_axis_idx(self.variable_or_axes, data_config.axes)
        except ValueError:
            try:
                reverse_axis_idx, slc = self._find_axis_idx(
                    self.variable_or_axes, data_config.axes[::-1]
                )
                axis_idx = -1 - reverse_axis_idx
            except ValueError as exc:
                raise ValueError(f"Axis {self.variable_or_axes} not found in data \
                        config {data_config}.") from exc
        return axis_idx, slc

    def _find_axis_idx(
        self, variable_or_axis, axes: list[Axes]
    ) -> tuple[int | slice, slice | None]:
        counter = 0
        for i_axis in axes:
            if isinstance(i_axis, EllipsisAxes):
                raise ValueError
            if any(isinstance(i_dim, EllipsisDim) for i_dim in i_axis.shape):
                raise ValueError

            if i_axis is variable_or_axis:

                if len(i_axis.shape) == 1:
                    return counter, None
                else:
                    return slice(counter, counter + len(i_axis.shape)), None
            if isinstance(variable_or_axis, Variable):
                if isinstance(i_axis, (FeatureAxes, GeometryAxes)):
                    i_var = i_axis.variables
                    if variable_or_axis in i_var:
                        if len(i_axis.shape) == 1:
                            return counter
                        return counter, i_var.get_slice(variable_or_axis)

            counter += len(i_axis.shape)

        raise ValueError
