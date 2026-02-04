from __future__ import annotations
from types import EllipsisType

from .axis import Axis, FeatureAxis, BatchAxis
from .variables import Variable


class DataConfiguration:
    """
    sets the basic type (numpy array, torch tensor etc) and shape of the data,
    and also collections of these will be used to check compatibility of the algorithms
    also include variables and their names?

    -> later implement several configuration conversion methods (and visualization),
    it should be possible to this during the execution of an algorithm as well as offline
    ->  also suggest automatic conversion methods between compatible configurations

    TODO: how to handle dictionaries, lists etc... nested structures?
    -> Best to do this in the dataset class? Since here we only specify the general
    shape of the data (axis.size == None, means variable size along that axis).

    """

    def __init__(
        self,
        dtype,
        axes: list[Axis | EllipsisType],
        feature_axis: FeatureAxis | EllipsisType,
        connection_to_axes: dict[Variable, list[Axis]] | None = None,
    ):
        assert feature_axis in axes, "Feature axis must be one of the axes."
        self.dtype = dtype  # TODO: Currently None if type does not matter?
        self.axes = axes
        self.feature_axis = feature_axis
        self.connection_to_axes = (
            connection_to_axes if connection_to_axes is not None else {}
        )
        self.batch_axis_idx: int | None = None

    @property
    def batch_axis(self) -> int:
        if self.batch_axis_idx is not None:
            return self.batch_axis_idx
        for i, axis in enumerate(self.axes):
            if isinstance(axis, BatchAxis):
                self.batch_axis_idx = i
                return self.batch_axis_idx
        raise ValueError("Data configuration has no batch axis.")

    def fits(self, other_config: DataConfiguration) -> bool:
        """Checks if another data configuration is compatible with this one.
        Meaning that the other configuration could be a specialization of this one,
        where some ellipsis are replaced by concrete axes or where the variables
        in the feature axis have been reduced.

        TODO: How is the default config in algo defined? Do we just have
        [Batch, ..., Feature] for example, but what is the feature axis, like where
        is it defined exactly? How do we compare it with the feature axis of the data?
        Because the algorithm can not now what axis the user names "features"? Or
        do we make this Feature axis always part of the configuration?
        """
        idx_self = 0
        idx_other = 0

        while idx_self < len(self.axes) and idx_other < len(other_config.axes):
            if self.axes[idx_self] is ...:
                # Skip ellipsis
                idx_self += 1
                if idx_self == len(self.axes):
                    # Trailing ellipsis matches everything remaining
                    break

                # Advance other_config.axes until we find the next self.axes element
                while (
                    idx_other < len(other_config.axes)
                    and other_config.axes[idx_other] != self.axes[idx_self]
                ):
                    idx_other += 1
            else:
                if other_config.axes[idx_other] != self.axes[idx_self]:
                    return False
                idx_self += 1
                idx_other += 1

        # Consume remaining ellipsis in self.axes
        while idx_self < len(self.axes) and (
            self.axes[idx_self] is ... or idx_self == len(self.axes) - 1
        ):
            idx_self += 1

        if not (idx_self == len(self.axes) and idx_other == len(other_config.axes)):
            return False

        # Check if variables in feature axis are compatible (or subset)
        if (
            other_config.feature_axis is ...
            or other_config.feature_axis.variables is None
            or self.feature_axis is ...
            or self.feature_axis.variables is None
        ):
            return True
        return other_config.feature_axis.variables in self.feature_axis.variables

    def __getitem__(self, key: int | slice | Variable) -> DataConfiguration:
        """Slice the configuration by axis index/indices or by Variables,
        to quickly obtain a new configuration.
        """
        if isinstance(key, Variable):
            if self.feature_axis is ... or self.feature_axis.variables is None:
                raise ValueError(
                    "Cannot slice by Variable when feature_axis is Ellipsis or "
                    "has no variables."
                )
            assert (
                key in self.feature_axis.variables
            ), "Variable slice must be a subset of the feature axis variables"
            # Create new axis with reduced variables
            new_feature_axis = FeatureAxis(size=key.dim, variables=key)
            return type(self)(
                self.dtype, self.axes, new_feature_axis, self.connection_to_axes
            )

        if isinstance(key, (int, slice)):
            raw = self.axes[key]

            sliced_axes: list[Axis | EllipsisType]
            if isinstance(raw, list):
                sliced_axes = raw
            else:
                sliced_axes = [raw]

            if len(sliced_axes) == 0:
                raise ValueError("Slice results in empty axes list")

            if self.feature_axis in sliced_axes:
                feature_axis = self.feature_axis
            else:
                feature_axis = ...

            return type(self)(
                self.dtype, sliced_axes, feature_axis, self.connection_to_axes
            )

        raise TypeError(f"Unsupported slicing type: {type(key)}")

    def __eq__(self, other_config: object) -> bool:
        if not isinstance(other_config, DataConfiguration):
            return False
        if len(other_config.axes) != len(self.axes):
            return False
        if self.dtype != other_config.dtype:
            if self.dtype is not None and other_config.dtype is not None:
                return False
        for i, other_axis in enumerate(other_config.axes):
            if not other_axis == self.axes[i]:
                return False
        return True

    def axes_of(self, var: Variable) -> list[Axis]:
        return self.connection_to_axes.get(var, [])

    def variables_on_axis(self, axis: Axis) -> Variable | None:
        for v, axes in self.connection_to_axes.items():
            if axis in axes:
                return v
        return None

    def map_variable_to_axes(self, var: Variable, axes: list[Axis]):
        for axis in axes:
            assert axis in self.axes, "All axes must be part of the configuration."
        self.connection_to_axes[var] = axes
