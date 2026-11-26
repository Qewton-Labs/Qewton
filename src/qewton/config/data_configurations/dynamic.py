"""Configurations at a place of a matching:
:class:`DynamicDataConfiguration`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qewton.config.axes import (
    AddedDim,
    Axes,
    AxesDim,
    BroadcastDim,
    SequenceProductDim,
    WindowDim,
)
from qewton.config.errors import DataConfigMismatchError, StructureUnknown

from .base import DataConfiguration

if TYPE_CHECKING:
    from qewton.config.matching import Matching


class DynamicDataConfiguration(DataConfiguration):
    """The configuration of a port at a place of a
    :class:`~qewton.config.matching.Matching`, e.g. of a node in a graph: an
    instance of its declaration that matching makes more concrete.

    Merging (:meth:`merge`) and the shape operations work through the
    matching: what they fix (e.g. that a dim is 1) is kept there, and their
    results belong to it; :meth:`resolve` shows what is known so far. Node
    authors get configurations of this kind in
    :meth:`~qewton.graphs.nodes.Node.relate`.

    Args:
        *axes (Axes | EllipsisAxes): The axes that describe the structure of
            the data.
        matching: The matching this configuration belongs to.
        dtype (optional): The datatype; None leaves it open.
        element_dtype (optional): The dtype of the elements of a tensor;
            None leaves it open.

    Attributes:
        matching (Matching): The matching this configuration belongs to.
    """

    def __init__(self, *axes, matching: Matching, dtype=None, element_dtype=None):
        super().__init__(*axes, dtype=dtype, element_dtype=element_dtype)
        self.matching = matching

    def merge(self, other: DataConfiguration) -> None:
        """Make this configuration and `other` describe the same data in the
        matching (see :meth:`Matching.merge_configurations`): corresponding
        axes and dims become one group, ellipses are bound to what they stand
        for, the dtypes are merged. Call :meth:`resolve` to see the result.

        Args:
            other: The configuration to merge with, of the matching or a
                declaration.

        Raises:
            DataConfigMismatchError: If the configurations contradict.
        """
        self.matching.merge_configurations(self, other)

    def resolve(self) -> DynamicDataConfiguration:
        """This configuration as known so far: merged objects replaced by
        their representatives, bound ellipses by what they stand for (see
        :meth:`Matching.resolve`).

        Returns:
            DynamicDataConfiguration: The resolved configuration.
        """
        return self.matching.resolve(self)

    def belongs_to(self, matching) -> bool:
        return matching is self.matching


    # region: Shape operations
    # Each operation returns the configuration of the result, built from the
    # objects of this one; indices count the flat dims, as for tensors. What an
    # operation fixes (e.g. that a dim is 1) goes through the matching of this
    # configuration, as does refining unknown structure by new unknowns where an
    # index needs it (see `Matching.refining`; otherwise `StructureUnknown`).

    def dim(self, index: int) -> AxesDim:
        """The dim at `index`, e.g. to relate it to a dim of another input."""
        slots = self._refined(index)
        return slots[_position(index, slots)][1]

    def merge_dim(self, index: int, dim: AxesDim) -> None:
        """Make the dim at `index` one with `dim`, e.g. to relate it to a
        size of the node (see :meth:`merge`)."""
        slots = self._refined(index)
        self.matching.merge(slots[_position(index, slots)][1], dim)

    def drop(self, index: int) -> DynamicDataConfiguration:
        """The configuration without the dim at `index`, e.g. after a sum over
        it without keeping it."""
        slots = self._refined(index)
        position = _position(index, slots)
        return self._from_slots(slots[:position] + slots[position + 1 :])

    def squeeze(self, index: int) -> DynamicDataConfiguration:
        """The configuration without the dim at `index`, which has size 1."""
        slots = self._refined(index)
        self.matching.merge(slots[_position(index, slots)][1], AxesDim(1))
        return self.drop(index)

    def set_size(self, index: int, size: int | AxesDim) -> DynamicDataConfiguration:
        """The configuration with the dim at `index` of `size`, e.g. 1 after a
        sum over it keeping it."""
        slots = self._refined(index)
        position = _position(index, slots)
        dim = size if isinstance(size, AxesDim) else AxesDim(size)
        return self._from_slots(
            slots[:position] + [(slots[position][0], dim)] + slots[position + 1 :]
        )

    def unsqueeze(self, index: int) -> DynamicDataConfiguration:
        """The configuration with a new dim of size 1 at `index` of the result."""
        if index >= 0:
            slots = self._refined(index - 1) if index > 0 else self._slots()
            position = index
        else:
            slots = self._refined(index + 1) if index < -1 else self._slots()
            position = len(slots) + index + 1
        return self._from_slots(slots[:position] + [(None, AxesDim(1))] + slots[position:])

    def permute(self, order: list[int]) -> DynamicDataConfiguration:
        """The configuration with its dims in `order`, which also fixes their
        number."""
        slots = self._exactly(len(order))
        return self._from_slots([slots[i] for i in order])

    def flatten(self, start: int = 0, end: int = -1) -> DynamicDataConfiguration:
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

    def unflatten(self, index: int, sizes: tuple[int, ...]) -> DynamicDataConfiguration:
        """The configuration with the dim at `index` split into dims of
        `sizes`; one of them may be -1."""
        slots = self._refined(index)
        position = _position(index, slots)
        axes, dim = slots[position]
        new_dims = [AxesDim() if size == -1 else AxesDim(size) for size in sizes]
        self.matching.merge(dim, SequenceProductDim(tuple(new_dims)))
        new_slots = [(axes, new_dim) for new_dim in new_dims]
        return self._from_slots(slots[:position] + new_slots + slots[position + 1 :])

    def reshape(self, sizes: tuple[int, ...]) -> DynamicDataConfiguration:
        """The configuration of the data reshaped to `sizes`; one of them may be
        -1. The new dims belong to axes of unknown kind."""
        items = tuple(self.axes)
        new_dims = [AxesDim() if size == -1 else AxesDim(size) for size in sizes]
        self.matching.merge(
            SequenceProductDim(items), SequenceProductDim(tuple(new_dims))
        )
        return self._from_slots([(None, new_dim) for new_dim in new_dims])

    def concat(self, others: list[DataConfiguration], index: int) -> DynamicDataConfiguration:
        """The configuration of this data and `others` joined along the dim at
        `index`, whose size is their sum; all other dims are merged."""
        matching = self.matching
        parts = [config._split_at(index) for config in [self, *others]]
        before, (axes, total), after = parts[0]
        for other_before, (_, dim), other_after in parts[1:]:
            matching.merge_configurations(
                _from_axes_slots(before, matching), _from_axes_slots(other_before, matching)
            )
            matching.merge_configurations(
                _from_axes_slots(after, matching), _from_axes_slots(other_after, matching)
            )
            total = AddedDim(total, dim)
        return self._from_slots(before + [(axes, total)] + after)

    def split(self, index: int, sizes: list[int]) -> list[DynamicDataConfiguration]:
        """The configurations of the pieces of this data split along the dim at
        `index` into `sizes`, whose sum is that dim."""
        before, (axes, dim), after = self._split_at(index)
        new_dims = [AxesDim(size) for size in sizes]
        total = new_dims[0]
        for new_dim in new_dims[1:]:
            total = AddedDim(total, new_dim)
        self.matching.merge(dim, total)
        return [self._from_slots(before + [(axes, new_dim)] + after) for new_dim in new_dims]

    def broadcast(self, other: DataConfiguration) -> DynamicDataConfiguration:
        """The configuration that this data and `other` broadcast to, aligned
        from the last dim; where one has size 1, the other's dim (and axes)
        is taken. Its dtype is left open.

        Raises:
            StructureUnknown: While the structure of either is unknown.
        """
        matching = self.matching
        mine, theirs = self._slots(), other._slots()
        if any(_is_open(slot) for slot in mine + theirs):
            raise StructureUnknown(f"{self} and {other} can not be broadcast yet.")
        common = min(len(mine), len(theirs))
        longer = mine if len(mine) >= len(theirs) else theirs
        aligned = list(zip(mine[len(mine) - common :], theirs[len(theirs) - common :]))
        result = longer[: len(longer) - common] + _broadcast_slots(aligned, matching)
        # The dtype of the result follows the backend's type promotion, so it
        # is left to the declaration of the output.
        return _from_axes_slots(result, matching)

    def window(
        self, index: int, kernel: int, stride: int = 1, padding: int = 0, dilation: int = 1
    ) -> DynamicDataConfiguration:
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
        expand = self._expand
        slots = []
        for axes in expand(self.axes):
            if axes.is_ellipsis:
                slots.append((axes, None))
            else:
                slots.extend((axes, dim) for dim in expand(axes.shape))
        return slots

    def _expand(self, items) -> list:
        """`items` as known to the matching."""
        return self.matching.expand(items)

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
                    axes, dim = open_slot
                    self.matching.open_up(axes if dim is None else dim, at_start)
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
                self.matching.bind(slot[0] if slot[1] is None else slot[1], ())
        slots = self._slots()
        if len(slots) != count:
            raise DataConfigMismatchError(f"{self} does not have {count} dims.")
        return slots

    def _split_at(self, index: int) -> tuple[list, tuple, list]:
        slots = self._refined(index)
        position = _position(index, slots)
        return slots[:position], slots[position], slots[position + 1 :]

    def _from_slots(self, slots: list[tuple]) -> DynamicDataConfiguration:
        """A configuration of the axes of `slots`: an axes whose dims are all
        kept in order is the same object, one that changed is a new axes of
        its kind, and a new dim without axes gets an axes of unknown kind."""
        axes = _from_axes_slots(slots, self.matching).axes
        return DynamicDataConfiguration(
            *axes, matching=self.matching, dtype=self.dtype, element_dtype=self.element_dtype
        )

    # endregion


def _position(index: int, slots: list) -> int:
    return index if index >= 0 else len(slots) + index


def _is_open(slot: tuple) -> bool:
    """Whether a slot stands for unknown structure (an unbound ellipsis)."""
    _, dim = slot
    return dim is None or dim.is_ellipsis


def _broadcast_slots(aligned: list, matching) -> list:
    """The slots that aligned pairs of slots broadcast to. Each dim is the
    other one where one has size 1, else a dim of both. Axes that share
    aligned slots form a group whose dims end up in one axes: the one that
    holds all of them (of the input all dims come from, if both do; of known
    kind before unknown), so that no axes is split."""
    # Per slot: the dim of the first, of the second, of either (equal sizes)
    # or of both (a broadcast dim).
    takes = []
    for (_, mine), (_, theirs) in aligned:
        my_size, their_size = matching.size(mine), matching.size(theirs)
        if their_size is not None and their_size == my_size:
            takes.append("either")
        elif their_size == 1:
            takes.append("first")
        elif my_size == 1:
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


def _from_axes_slots(slots: list, matching) -> DynamicDataConfiguration:
    """A configuration of the axes of `slots` in `matching`, see
    `DynamicDataConfiguration._from_slots`."""
    expand = matching.expand
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
    return DynamicDataConfiguration(*axes_list, matching=matching)
