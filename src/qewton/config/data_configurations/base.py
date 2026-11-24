from __future__ import annotations

import math
from typing import Any
from types import EllipsisType

from qewton.config.axes import (
    Axes,
    EllipsisAxes,
    EllipsisDim,
    GeometryAxes,
    FeatureAxes,
    AxesDim,
)
from qewton.config.dtypes import loaded_dtype, saved_dtype
from qewton.config.saving.loading import Deserializer
from qewton.config.saving.schema_keys import SavingKeys
from qewton.config.variables import Variable
from qewton.config.saving.saving import Serializable, Serializer



class DataConfiguration(Serializable):
    """A *DataConfiguration* describes the expected structure of the data,
    including the axes and their dimensions, as well as the data type.
    It is used to ensure that the data being passed through the graph
    matches the expected format for different methods/algorithms.

    Args:
        *axes (Axes | EllipsisAxes): The axes that describe the
            structure of the data.
        dtype (_type_, optional): The datatype used in this
            configuration. Defaults to None.
        element_dtype (optional): The dtype of the elements of a tensor, a
            marker of :mod:`qewton.config.dtypes` such as `Float32`; None
            leaves it open.

    A configuration as such is a declaration, e.g. of a port, and never
    changes; matching works on its instances in a
    :class:`~qewton.config.matching.Matching`
    (:class:`~qewton.config.data_configurations.dynamic.DynamicDataConfiguration`).
    """

    #: The dtype of the elements of a tensor; None leaves it open.
    element_dtype: Any = None

    def __init__(self, *axes: Axes | EllipsisAxes, dtype=None, element_dtype=None):
        self.axes = axes
        self.dtype = dtype
        self.element_dtype = element_dtype
        assert (
            len([f for f in axes if isinstance(f, FeatureAxes)]) <= 1
        ), "A DataConfig can have at most one FeatureAxes."

    def replace(self, axes: Axes, new_axes: list[Axes]) -> "DataConfiguration":
        """A configuration with `axes` replaced by `new_axes`, describing
        the same data reshaped: ``data.reshape(new.shape)``.

        Args:
            axes: Axes of this configuration.
            new_axes: Axes with as many elements as `axes`, in row-major
                order.

        Returns:
            DataConfiguration: The new configuration, with the same dtype.

        Raises:
            ValueError: If `axes` is not part of this configuration or the
                number of elements differs.
        """
        if not any(a is axes for a in self.axes):
            raise ValueError(f"{axes} is not part of the data configuration {self}.")

        def size(axes_list):
            return math.prod(d.size for a in axes_list for d in a.shape)

        if size(new_axes) != size([axes]):
            raise ValueError(
                f"{axes} has {size([axes])} elements, the new axes have "
                f"{size(new_axes)}."
            )
        replaced = [n for a in self.axes for n in (new_axes if a is axes else [a])]
        return DataConfiguration(*replaced, dtype=self.dtype)

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

    @property
    def shape(self) -> tuple[int | None | EllipsisType, ...]:
        shape_list = []
        for axes in self.axes:
            if isinstance(axes, EllipsisAxes):
                shape_list.append(Ellipsis)
            else:
                for dim in axes.shape:
                    if isinstance(dim, EllipsisDim):
                        shape_list.append(Ellipsis)
                    elif isinstance(dim, AxesDim):
                        shape_list.append(dim.size)
                    else:
                        shape_list.append(dim)
        return tuple(shape_list)

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

    @property
    def geometry_axes(self) -> GeometryAxes | list[GeometryAxes] | None:
        """Returns the geometry axes of this configuration.

        Returns:
            GeometryAxes | list[GeometryAxes] | None: The geometry axes if
                defined, otherwise None.
        """
        geom_axes = []
        for axes in self.axes:
            if isinstance(axes, GeometryAxes):
                geom_axes.append(axes)
        if len(geom_axes) == 1:
            return geom_axes[0]
        if len(geom_axes) > 1:
            return geom_axes
        return None

    def belongs_to(self, matching) -> bool:
        """Whether this configuration is an instance in `matching`; a
        declaration belongs to none."""
        return False

    def set_dtype(self, new_dtype):
        self.dtype = new_dtype

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

    @classmethod
    def of_value(cls, value: Any) -> DataConfiguration | None:
        """The configuration of a given value, e.g. the default of an
        unconnected input: none for a Python scalar, axes of unknown kind for
        an array; None if nothing is known about it."""
        if isinstance(value, (bool, int, float, complex)):
            return DataConfiguration(dtype=type(value))
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        return DataConfiguration(*(Axes(AxesDim(int(size))) for size in shape))


    def axes_of(self, reference: Variable | Axes) -> Axes:
        """Find the axes of this configuration that hold a reference.

        Args:
            reference: An Axes of this configuration, or a Variable (or a
                part of one) of a GeometryAxes' geometry or of the
                FeatureAxes.

        Returns:
            Axes: `reference` itself if it is an Axes, otherwise the
                FeatureAxes or GeometryAxes whose variable contains it. A
                Variable held by both, e.g. the coordinates of a geometry
                given as data, lies in the FeatureAxes: the data holds its
                values.

        Raises:
            ValueError: If no axes of this configuration hold `reference`.
        """
        if any(axes is reference for axes in self.axes):
            return reference
        if isinstance(reference, Variable):
            candidates = sorted(self.axes, key=lambda a: not isinstance(a, FeatureAxes))
            for axes in candidates:
                if isinstance(axes, (GeometryAxes, FeatureAxes)):
                    variable = axes.variables
                    if variable is not None and reference in variable:
                        return axes
        name = reference.name if isinstance(reference, Variable) else reference
        raise ValueError(
            f"{name} is not part of the data: it holds {self.describe()}. Bind one "
            "of these, e.g. a feature or the Variable of a geometry."
        )

    def describe(self) -> str:
        """What the axes of this configuration hold, by name, e.g.
        ``"batch axes (50), the geometry of t (50), the features u, p"``.

        Returns:
            str: One part per axes.
        """
        parts = []
        for axes in self.axes:
            sizes = ", ".join(str(d.size) for d in axes.shape) if not isinstance(
                axes, EllipsisAxes
            ) else "..."
            if isinstance(axes, FeatureAxes):
                variable = axes.variables
                names = (
                    ", ".join(q.name for q in variable.quantities)
                    if variable is not None and not variable.is_empty
                    else None
                )
                parts.append(f"the features {names}" if names else f"unnamed features ({sizes})")
            elif isinstance(axes, GeometryAxes):
                variable = axes.geometry.variable
                named = variable is not None and not variable.is_empty and variable.name
                parts.append(
                    f"the geometry of {variable.name} ({sizes})"
                    if named
                    else f"a geometry without a Variable ({sizes})"
                )
            else:
                parts.append(f"{type(axes).__name__.removesuffix('Axes').lower()} axes ({sizes})")
        return ", ".join(parts)

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
                return slice(counter, counter + len(i_axis.shape)), None
            if isinstance(variable_or_axis, Variable):
                if isinstance(i_axis, (FeatureAxes, GeometryAxes)):
                    i_var = i_axis.variables
                    if variable_or_axis in i_var:
                        if len(i_axis.shape) == 1:
                            return counter, None
                        return counter, i_var.get_slice(variable_or_axis)

            counter += len(i_axis.shape)

        raise ValueError

    def save(self, serializer: Serializer) -> None:
        axes_idx, dtype_idx, element_idx = serializer.add_objects(
            self, [self.axes, saved_dtype(self.dtype), saved_dtype(self.element_dtype)]
        )
        data_config = {
            SavingKeys.KEY_TYPE: SavingKeys.KEY_SERIALIZABLE,
            SavingKeys.KEY_CLASS: self.__class__.__name__,
            SavingKeys.KEY_MODULE: self.__class__.__module__,
            "axes": axes_idx,
            "dtype": dtype_idx,
            "element_dtype": element_idx,
        }
        serializer.set_serialization_data(id(self), data_config)

    def load(self, serializer: Deserializer, data_config: dict) -> None:
        self.axes = serializer.id_to_obj[data_config["axes"]]
        if serializer.version < 2:
            # Version 1 saved dtypes as text; they are left open.
            self.dtype = self.element_dtype = None
            return
        self.dtype = loaded_dtype(serializer.id_to_obj[data_config["dtype"]])
        self.element_dtype = loaded_dtype(serializer.id_to_obj[data_config["element_dtype"]])

    def ready_to_load(self, serializer: Deserializer, data_config: dict) -> bool:
        keys = ("axes",) if serializer.version < 2 else ("axes", "dtype", "element_dtype")
        return all(serializer.ready_to_reference.get(data_config[key], False) for key in keys)
