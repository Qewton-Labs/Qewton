from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from types import EllipsisType

from qewton.config.axes import (
    AddedDim,
    BroadcastDim,
    SequenceProductDim,
    WindowDim,
    Axes,
    EllipsisAxes,
    EllipsisDim,
    GeometryAxes,
    FeatureAxes,
    AxesDim,
)
from qewton.config.dtypes import loaded_dtype, merge_dtypes, merge_element_dtypes, saved_dtype
from qewton.config.errors import DataConfigMismatchError, StructureUnknown
from qewton.config.sequences import bind, expand, merge_sequences
from qewton.config.saving.loading import Deserializer
from qewton.config.saving.schema_keys import SavingKeys
from qewton.config.variables import Variable
from qewton.config.saving.saving import Serializable, Serializer

## TODO: could we simplify the config passing to use common AxesDim
# objects nearly around the whole graph? this would allow for less
# passing operations and less objects.


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

    def merge(self, other: DataConfiguration) -> None:
        """Make this configuration and `other` describe the same data.

        Merges the dtypes and the axes: corresponding axes and dims become
        one object, ellipses are bound to what they stand for. Call
        :meth:`resolve` to see the result in the axes of both.

        Args:
            other: The configuration to merge with.

        Raises:
            DataConfigMismatchError: If the configurations contradict.
        """
        self.dtype = other.dtype = merge_dtypes(self.dtype, other.dtype)
        self.element_dtype = other.element_dtype = merge_element_dtypes(
            self.element_dtype, other.element_dtype
        )
        merge_sequences(self.axes, other.axes)

    def resolve(self) -> DataConfiguration:
        """Replace every merged axes and dim by its representative and every
        bound ellipsis by what it stands for, in place.

        Returns:
            DataConfiguration: This configuration.
        """
        self.axes = tuple(expand(self.axes))
        for axes in self.axes:
            axes._shape = tuple(expand(axes._shape))
        return self

    def mark_origin(self, origin: Any) -> None:
        """Record `origin` (e.g. a port) as what brought in every known size
        of this configuration that has no origin yet; errors name it."""
        for axes in expand(self.axes):
            for dim in expand(axes.shape):
                if dim.size is not None and dim.origin is None:
                    dim.origin = origin

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

    # region: Shape operations
    # Each operation returns the configuration of the result, built from the
    # objects of this one; indices count the flat dims, as for tensors. Unknown
    # structure is refined by new unknowns where an index needs it, unless
    # refinement is switched off (see `refining`): then `StructureUnknown`.

    def dim(self, index: int) -> AxesDim:
        """The dim at `index`, e.g. to relate it to a dim of another input."""
        slots = self._refined(index)
        return slots[_position(index, slots)][1]

    def drop(self, index: int) -> DataConfiguration:
        """The configuration without the dim at `index`, e.g. after a sum over
        it without keeping it."""
        slots = self._refined(index)
        position = _position(index, slots)
        return self._from_slots(slots[:position] + slots[position + 1 :])

    def squeeze(self, index: int) -> DataConfiguration:
        """The configuration without the dim at `index`, which has size 1."""
        slots = self._refined(index)
        slots[_position(index, slots)][1].merge(AxesDim(1))
        return self.drop(index)

    def set_size(self, index: int, size: int | AxesDim) -> DataConfiguration:
        """The configuration with the dim at `index` of `size`, e.g. 1 after a
        sum over it keeping it."""
        slots = self._refined(index)
        position = _position(index, slots)
        dim = size if isinstance(size, AxesDim) else AxesDim(size)
        return self._from_slots(
            slots[:position] + [(slots[position][0], dim)] + slots[position + 1 :]
        )

    def unsqueeze(self, index: int) -> DataConfiguration:
        """The configuration with a new dim of size 1 at `index` of the result."""
        if index >= 0:
            slots = self._refined(index - 1) if index > 0 else self._slots()
            position = index
        else:
            slots = self._refined(index + 1) if index < -1 else self._slots()
            position = len(slots) + index + 1
        return self._from_slots(slots[:position] + [(None, AxesDim(1))] + slots[position:])

    def permute(self, order: list[int]) -> DataConfiguration:
        """The configuration with its dims in `order`, which also fixes their
        number."""
        slots = self._exactly(len(order))
        return self._from_slots([slots[i] for i in order])

    def flatten(self, start: int = 0, end: int = -1) -> DataConfiguration:
        """The configuration with the dims from `start` to `end` merged into
        one, whose size is their product."""
        bounds = [start - 1] if start > 0 else []
        if end >= 0 or end < -1:
            bounds.append(end if end >= 0 else end + 1)
        slots = self._refined(*bounds) if bounds else self._slots()
        first = _position(start, slots)
        last = end if end >= 0 else len(slots) + end
        items = [axes if dim is None else dim for axes, dim in slots[first : last + 1]]
        product = (None, SequenceProductDim(tuple(items)))
        return self._from_slots(slots[:first] + [product] + slots[last + 1 :])

    def unflatten(self, index: int, sizes: tuple[int, ...]) -> DataConfiguration:
        """The configuration with the dim at `index` split into dims of
        `sizes`; one of them may be -1."""
        slots = self._refined(index)
        position = _position(index, slots)
        axes, dim = slots[position]
        new_dims = [AxesDim() if size == -1 else AxesDim(size) for size in sizes]
        dim.merge(SequenceProductDim(tuple(new_dims)))
        new_slots = [(axes, new_dim) for new_dim in new_dims]
        return self._from_slots(slots[:position] + new_slots + slots[position + 1 :])

    def reshape(self, sizes: tuple[int, ...]) -> DataConfiguration:
        """The configuration of the data reshaped to `sizes`; one of them may be
        -1. The new dims belong to axes of unknown kind."""
        items = tuple(self.axes)
        new_dims = [AxesDim() if size == -1 else AxesDim(size) for size in sizes]
        SequenceProductDim(items).merge(SequenceProductDim(tuple(new_dims)))
        return self._from_slots([(None, new_dim) for new_dim in new_dims])

    def concat(self, others: list[DataConfiguration], index: int) -> DataConfiguration:
        """The configuration of this data and `others` joined along the dim at
        `index`, whose size is their sum; all other dims are merged."""
        parts = [config._split_at(index) for config in [self, *others]]
        before, (axes, total), after = parts[0]
        for other_before, (_, dim), other_after in parts[1:]:
            _from_axes_slots(before).merge(_from_axes_slots(other_before))
            _from_axes_slots(after).merge(_from_axes_slots(other_after))
            total = AddedDim(total, dim)
        return self._from_slots(before + [(axes, total)] + after)

    def split(self, index: int, sizes: list[int]) -> list[DataConfiguration]:
        """The configurations of the pieces of this data split along the dim at
        `index` into `sizes`, whose sum is that dim."""
        before, (axes, dim), after = self._split_at(index)
        new_dims = [AxesDim(size) for size in sizes]
        total = new_dims[0]
        for new_dim in new_dims[1:]:
            total = AddedDim(total, new_dim)
        dim.merge(total)
        return [self._from_slots(before + [(axes, new_dim)] + after) for new_dim in new_dims]

    def broadcast(self, other: DataConfiguration) -> DataConfiguration:
        """The configuration that this data and `other` broadcast to, aligned
        from the last dim; where one has size 1, the other's dim (and axes)
        is taken. Its dtype is left open.

        Raises:
            StructureUnknown: While the structure of either is unknown.
        """
        mine, theirs = self._slots(), other._slots()
        if any(_is_open(slot) for slot in mine + theirs):
            raise StructureUnknown(f"{self} and {other} can not be broadcast yet.")
        common = min(len(mine), len(theirs))
        longer = mine if len(mine) >= len(theirs) else theirs
        aligned = list(zip(mine[len(mine) - common :], theirs[len(theirs) - common :]))
        result = longer[: len(longer) - common] + _broadcast_slots(aligned)
        # The dtype of the result follows the backend's type promotion, so it
        # is left to the declaration of the output.
        return _from_axes_slots(result)

    def window(
        self, index: int, kernel: int, stride: int = 1, padding: int = 0, dilation: int = 1
    ) -> DataConfiguration:
        """The configuration after a sliding window (a convolution or pooling)
        over the dim at `index`."""
        slots = self._refined(index)
        position = _position(index, slots)
        axes, dim = slots[position]
        new_dim = WindowDim(dim, kernel, stride, padding, dilation)
        return self._from_slots(slots[:position] + [(axes, new_dim)] + slots[position + 1 :])

    def _slots(self) -> list[tuple]:
        """(axes, dim) for every dim in order; an unbound ellipsis of axes is
        (ellipsis, None)."""
        slots = []
        for axes in expand(self.axes):
            if axes.is_ellipsis:
                slots.append((axes, None))
            else:
                slots.extend((axes, dim) for dim in expand(axes.shape))
        return slots

    def _refined(self, *indices: int) -> list[tuple]:
        """The slots, after binding ellipses so that every slot from the start
        (or end, for a negative index) up to each index is a dim."""
        front = max([i + 1 for i in indices if i >= 0], default=0)
        back = max([-i for i in indices if i < 0], default=0)
        while True:
            slots = self._slots()
            ends = [(slots[:front], True), (slots[len(slots) - back :][::-1] if back else [], False)]
            opened = False
            for part, at_start in ends:
                open_slot = next((slot for slot in part if _is_open(slot)), None)
                if open_slot is not None:
                    _open_up(open_slot, at_start)
                    opened = True
                    break
            if not opened:
                if len(slots) < max(front, back):
                    raise ValueError(f"{self} has no dim at index {max(front, back)}.")
                return slots

    def _exactly(self, count: int) -> list[tuple]:
        """The slots, after binding ellipses so that there are `count` dims."""
        slots = self._refined(count - 1) if count else self._slots()
        for slot in slots[count:]:
            if _is_open(slot):
                bind(slot[0] if slot[1] is None else slot[1], ())
        slots = self._slots()
        if len(slots) != count:
            raise DataConfigMismatchError(f"{self} does not have {count} dims.")
        return slots

    def _split_at(self, index: int) -> tuple[list, tuple, list]:
        slots = self._refined(index)
        position = _position(index, slots)
        return slots[:position], slots[position], slots[position + 1 :]

    def _from_slots(self, slots: list[tuple]) -> DataConfiguration:
        """A configuration of the axes of `slots`: an axes whose dims are all
        kept in order is the same object, one that changed is a new axes of
        its kind, and a new dim without axes gets an axes of unknown kind."""
        axes = _from_axes_slots(slots).axes
        return DataConfiguration(*axes, dtype=self.dtype, element_dtype=self.element_dtype)

    # endregion

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



def _position(index: int, slots: list) -> int:
    return index if index >= 0 else len(slots) + index


def _is_open(slot: tuple) -> bool:
    """Whether a slot stands for unknown structure (an unbound ellipsis)."""
    axes, dim = slot
    return dim is None or dim.is_ellipsis


#: Whether shape operations may refine unknown structure (see `refining`).
_REFINING: ContextVar[bool] = ContextVar("refining", default=True)


@contextmanager
def refining(allowed: bool):
    """Context in which shape operations refine unknown structure by new
    unknowns (`allowed`) or raise `StructureUnknown` instead."""
    token = _REFINING.set(allowed)
    try:
        yield
    finally:
        _REFINING.reset(token)


def _open_up(slot: tuple, at_start: bool) -> None:
    """Bind the ellipsis of `slot` to one new unknown and a new ellipsis for
    the rest, the unknown at the start or the end."""
    if not _REFINING.get():
        raise StructureUnknown("The structure behind an ellipsis is not known yet.")
    axes, dim = slot
    if dim is None:
        unit, rest, ellipsis = Axes(AxesDim()), EllipsisAxes(), axes
    else:
        unit, rest, ellipsis = AxesDim(), EllipsisDim(), dim
    bind(ellipsis, (unit, rest) if at_start else (rest, unit))


def _broadcast_slots(aligned: list) -> list:
    """The slots that aligned pairs of slots broadcast to. Each dim is the
    other one where one has size 1, else a dim of both. Axes that share
    aligned slots form a group whose dims end up in one axes: the one that
    holds all of them (of the input all dims come from, if both do; of known
    kind before unknown), so that no axes is split."""
    # Per slot: the dim of the first, of the second, of either (equal sizes)
    # or of both (a broadcast dim).
    takes = []
    for (_, mine), (_, theirs) in aligned:
        if theirs.size is not None and theirs.size == mine.size:
            takes.append("either")
        elif theirs.size == 1:
            takes.append("first")
        elif mine.size == 1:
            takes.append("second")
        else:
            takes.append("both")
    # Group the slots: two slots belong together if they share an axes.
    group = list(range(len(aligned)))

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    first_slot: dict = {}
    for i, ((my_axes, _), (their_axes, _)) in enumerate(aligned):
        for axes in (my_axes, their_axes):
            j = first_slot.setdefault(id(axes), i)
            group[root(i)] = root(j)
    members: dict = {}
    for i in range(len(aligned)):
        members.setdefault(root(i), []).append(i)
    chosen = {}
    for slots in members.values():
        mine = {id(aligned[i][0][0]) for i in slots}
        theirs = {id(aligned[i][1][0]) for i in slots}
        my_axes, their_axes = aligned[slots[0]][0][0], aligned[slots[0]][1][0]
        my_spans, their_spans = len(mine) == 1, len(theirs) == 1
        from_second = all(takes[i] in ("second", "either") for i in slots) and any(
            takes[i] == "second" for i in slots
        )
        if their_spans and (not my_spans or from_second):
            axes = their_axes
        elif my_spans and their_spans and type(my_axes) is Axes and type(their_axes) is not Axes:
            axes = their_axes
        else:
            axes = my_axes
        for i in slots:
            chosen[i] = axes
    slots = []
    for i, (((my_axes, mine), (their_axes, theirs)), take) in enumerate(zip(aligned, takes)):
        if take == "both":
            dim = BroadcastDim(mine, theirs)
        elif take == "second" or take == "either" and chosen[i] is their_axes:
            dim = theirs
        else:
            dim = mine
        slots.append((chosen[i], dim))
    return slots


def _from_axes_slots(slots: list) -> DataConfiguration:
    """A configuration of the axes of `slots`, see
    `DataConfiguration._from_slots`."""
    axes_list, source, dims = [], None, []

    def flush():
        if not dims:
            return
        if source is None:
            axes_list.extend(Axes(dim) for dim in dims)
        elif dims == expand(source.shape):
            axes_list.append(source)
        else:
            axes_list.append(source.with_dims(dims))

    for axes, dim in slots:
        if dim is None:
            flush()
            source, dims = None, []
            axes_list.append(axes)
        elif axes is source and axes is not None:
            dims.append(dim)
        else:
            flush()
            source, dims = axes, [dim]
    flush()
    return DataConfiguration(*axes_list)
