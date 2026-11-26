"""Matching of data configurations: :class:`Matching`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Hashable

from qewton.config.axes import Axes, AxesDim
from qewton.config.data_configurations import DynamicDataConfiguration
from qewton.config.errors import DataConfigMismatchError, StructureUnknown


class Matching:
    """The state of matching data configurations, e.g. of one graph.

    Declared configurations never change. At a *place* (e.g. a node in a
    graph) a declaration is instantiated: every object that can become more
    concrete there (an unknown dim, an ellipsis, axes of unknown kind or
    content, and what contains them) gets a stand-in of its own, every other
    object is used as it is (:meth:`place`). Merging makes objects one group
    with a *representative*, binds ellipses to sequences and relates sizes
    through operation dims; all of it is kept here, in tables over the
    objects.

    Configurations of a matching (its places, :meth:`resolve`) and the
    results of their shape operations are
    :class:`~qewton.config.data_configurations.DynamicDataConfiguration`\ s
    of it: what those operations fix (e.g. that a dim is 1) goes through it.

    Attributes:
        refining (bool): Whether shape operations may refine unknown
            structure by binding ellipses to new unknowns; else they raise
            :class:`~qewton.config.errors.StructureUnknown`.
    """

    def __init__(self) -> None:
        self.refining = True
        self._places: dict[Hashable, dict] = {}
        self._stand_ins: dict[Hashable, dict] = {}
        self._parent: dict = {}
        self._bound: dict = {}
        self._waiting: dict = {}
        self._dependents: dict = {}
        self._equal_operations: dict = {}
        self._found: dict = {}
        self._origin: dict = {}
        self._parameters: dict = {}
        self._entered: set = set()
        self._built: dict = {}

    # region: Places

    def place(self, key: Hashable, declarations: dict) -> dict:
        """Instantiate `declarations` at the place `key`; ports the place
        has already are kept.

        Args:
            key: The place, e.g. a node at a path of GraphNodes.
            declarations: The declared configuration of each port.

        Returns:
            dict: The configuration of every port at the place; an open
                object shared by declarations is one stand-in there.
        """
        configs = self._places.setdefault(key, {})
        stand_ins = self._stand_ins.setdefault(key, {})
        for port, declaration in declarations.items():
            if port not in configs:
                configs[port] = self._instance(declaration, port, stand_ins)
        return configs

    def has_place(self, key: Hashable) -> bool:
        """Whether `key` was instantiated."""
        return key in self._places

    def configs(self, key: Hashable) -> dict:
        """The configuration of every port at the place `key`."""
        return self._places[key]

    def places(self) -> dict:
        """The configurations of the ports of every place, by place."""
        return dict(self._places)

    def _instance(self, declaration, origin, stand_ins: dict) -> DynamicDataConfiguration:
        config = DynamicDataConfiguration(
            *(self._stand_in(axes, stand_ins) for axes in declaration.axes),
            matching=self,
            dtype=declaration.dtype,
            element_dtype=declaration.element_dtype,
        )
        self._mark_origin(config.axes, origin)
        return config

    def _stand_in(self, item, stand_ins: dict):
        """`item` itself if it is the same at every place; else its stand-in
        at this place, built around stand-ins of its open parts."""
        if not item.is_open:
            return item
        if item not in stand_ins:
            parts = tuple(self._stand_in(part, stand_ins) for part in item.parts)
            stand_ins[item] = item.with_parts(parts)
        return stand_ins[item]

    def _mark_origin(self, items, origin) -> None:
        for axes in self.expand(items):
            for dim in self.expand(axes.shape):
                representative = self.representative(dim)
                if self.size(dim) is not None and representative not in self._origin:
                    self._origin[representative] = origin

    # endregion
    # region: Groups and sizes

    def representative(self, item):
        """The object that stands for `item` and everything merged with it."""
        self._enter(item)
        parent = self._parent.get(item)
        if parent is None:
            return item
        representative = self.representative(parent)
        self._parent[item] = representative
        return representative

    def size(self, item) -> int | None:
        """The size of `item` (a dim; None for axes) as far as known: its
        own (declared, or computed from its operands), else the one found
        for its group."""
        representative = self.representative(item)
        size = representative.size_in(self)
        return size if size is not None else self._found.get(representative)

    def expand(self, items: Sequence) -> list:
        """The representatives of `items`, with every bound ellipsis replaced
        by the sequence it is bound to."""
        expanded = []
        for item in items:
            item = self.representative(item)
            if item.is_ellipsis and item in self._bound:
                expanded.extend(self.expand(self._bound[item]))
            else:
                expanded.append(item)
        return expanded

    def origin(self, dim) -> Any:
        """What brought in the size of `dim`, e.g. a port; None if unknown."""
        return self._origin.get(self.representative(dim))

    def _enter(self, item) -> None:
        """Register what `item` brings in when it is first met: its
        HyperParameters, the items its size depends on."""
        if item in self._entered:
            return
        self._entered.add(item)
        if item.parameters:
            self._parameters[item] = item.parameters
        self._watch(item)

    def _watch(self, item) -> None:
        """Be told when an item that `item` depends on changes."""
        for operand in item.depends_on(self):
            representative = self.representative(operand)
            if item not in self._dependents.get(representative, ()):
                self._dependents[representative] = self._dependents.get(
                    representative, ()
                ) + (item,)

    # endregion
    # region: Merging

    def merge(self, first, second):
        """Make two axes or two dims one group.

        Returns:
            The representative of both.

        Raises:
            DataConfigMismatchError: If they contradict.
        """
        return _for_kind(_MERGE, first)(self, first, second)

    def merge_configurations(self, first, second) -> None:
        """Make two configurations describe the same data: their dtypes and
        their sequences of axes. Only configurations of this matching are
        written to; a declaration is read only.

        Raises:
            DataConfigMismatchError: If they contradict.
        """
        from qewton.config.dtypes import merge_dtypes, merge_element_dtypes

        dtype = merge_dtypes(first.dtype, second.dtype)
        element_dtype = merge_element_dtypes(first.element_dtype, second.element_dtype)
        for config in (first, second):
            if config.belongs_to(self):
                config.dtype, config.element_dtype = dtype, element_dtype
        self.merge_sequences(first.axes, second.axes)

    def compatible(self, first, second) -> bool:
        """Whether two axes or two dims could be merged."""
        return _for_kind(_COMPATIBLE, first)(self, first, second)

    def _merge_axes(self, first, second):
        first, second = self.representative(first), self.representative(second)
        if first is second:
            return first
        if not first.kind_compatible(second) or not first.contents_agree(second):
            raise DataConfigMismatchError(f"{first} and {second} can not be one axes.")
        keep, gone = (second, first) if second.more_concrete_than(first) else (first, second)
        self._parent[gone] = keep
        self.merge_sequences(keep.shape, gone.shape)
        return keep

    def _axes_compatible(self, first, second) -> bool:
        first, second = self.representative(first), self.representative(second)
        return (
            first.kind_compatible(second)
            and first.contents_agree(second)
            and self.sequences_compatible(first.shape, second.shape)
        )

    def _merge_dims(self, first, second):
        first, second = self.representative(first), self.representative(second)
        if first is second:
            return first
        keep, gone = (second, first) if self._rank(second) > self._rank(first) else (first, second)
        if self._feeds(gone, keep):
            keep, gone = gone, keep
        sizes = (self.size(first), self.size(second))
        if None not in sizes and sizes[0] != sizes[1]:
            raise DataConfigMismatchError(self._contradiction(first, second))
        self._parent[gone] = keep
        if keep not in self._origin and gone in self._origin:
            self._origin[keep] = self._origin[gone]
        for table in (self._dependents, self._equal_operations, self._parameters):
            if gone in table:
                table[keep] = table.get(keep, ()) + table.pop(gone)
        if gone in self._found and keep not in self._found:
            self._found[keep] = self._found.pop(gone)
        if gone.is_relation:
            self._equal_operations[keep] = self._equal_operations.get(keep, ()) + (gone,)
        size = sizes[0] if sizes[0] is not None else sizes[1]
        if size is not None:
            self._settle(keep, size)
        return keep

    def _dims_compatible(self, first, second) -> bool:
        if self.representative(first).is_ellipsis or self.representative(second).is_ellipsis:
            return True
        mine, theirs = self.size(first), self.size(second)
        return mine is None or theirs is None or mine == theirs

    def _rank(self, dim) -> tuple:
        """Which dim of a group stands for it: an operation dim, as its
        relation may still solve for its operands, then a dim of known
        size."""
        return (dim.is_relation, self.size(dim) is not None)

    def _feeds(self, dim, operation) -> bool:
        """Whether `operation` is computed from `dim`, directly or through
        other operations; such an operation can not stand for `dim`."""
        stack, seen = [dim], set()
        while stack:
            for dependent in self._dependents.get(stack.pop(), ()):
                dependent = self.representative(dependent)
                if dependent is operation:
                    return True
                if dependent not in seen:
                    seen.add(dependent)
                    stack.append(dependent)
        return False

    def _settle(self, dim, size: int) -> None:
        """Give the group of `dim` its size and tell what depends on it."""
        own = dim.size_in(self)
        if own is not None and own != size:
            raise DataConfigMismatchError(
                f"{dim} is {own}, but {size} (from {self.origin(dim)}) is required."
            )
        if own is None:
            self._found[dim] = size
        for operand, operand_size in dim.solve(size, self):
            self._merge_dims(operand, AxesDim(operand_size))
        self._infer_parameters(dim, size)
        for operation in self._equal_operations.get(dim, ()):
            self._settle(operation, size)
        for dependent in self._dependents.get(dim, ()):
            self._operand_changed(dependent)

    def _operand_changed(self, operation) -> None:
        """Check `operation` again after an item it depends on changed."""
        self._watch(operation)
        representative = self.representative(operation)
        computed = operation.size_in(self)
        if representative is not operation:
            if computed is not None and self.size(representative) != computed:
                self._merge_dims(representative, AxesDim(computed))
            return
        if operation in self._found:
            self._settle(operation, self._found[operation])
        elif computed is not None:
            self._infer_parameters(operation, computed)
            for equal in self._equal_operations.get(operation, ()):
                self._settle(equal, computed)
            for dependent in self._dependents.get(operation, ()):
                self._operand_changed(dependent)

    def _infer_parameters(self, dim, size: int) -> None:
        """Tell the HyperParameters bound to the group of `dim` its size (see
        `HyperParameter.infer`)."""
        for parameter in self._parameters.get(dim, ()):
            parameter.infer(size)

    def _contradiction(self, first, second) -> str:
        """The message for two dims of different sizes that should be one."""

        def described(dim) -> str:
            origin = self.origin(dim)
            sources = [f"from {origin}"] if origin is not None else []
            sources += [
                f"HyperParameter {p.name!r}"
                for p in self._parameters.get(self.representative(dim), ())
            ]
            size = self.size(dim)
            return f"{size} ({', '.join(sources)})" if sources else str(size)

        return f"Sizes {described(first)} and {described(second)} do not match."

    # endregion
    # region: Sequences

    def merge_sequences(self, first: Sequence, second: Sequence) -> None:
        """Merge two sequences of axes or dims item by item; ellipses take
        the rest. Of the consistent alignments, those that merge the most are
        preferred (no new ellipsis between the sides); a single remaining one
        is applied, several wait until an ellipsis is bound, so the result
        does not depend on the order of merges.

        Raises:
            DataConfigMismatchError: If no alignment of the sequences is
                consistent.
        """
        xs, ys = self.expand(first), self.expand(second)
        start = self._merge_while_plain(xs, ys)
        end = self._merge_while_plain(xs[start:][::-1], ys[start:][::-1])
        middle_x, middle_y = xs[start : len(xs) - end], ys[start : len(ys) - end]
        self._merge_middles(middle_x, middle_y, (tuple(first), tuple(second)))

    def sequences_compatible(self, first: Sequence, second: Sequence) -> bool:
        """Whether two sequences could be merged; with an ellipsis, only
        their items outside of it are checked."""
        xs, ys = self.expand(first), self.expand(second)
        if any(item.is_ellipsis for item in xs + ys):
            return True
        return len(xs) == len(ys) and all(self.compatible(x, y) for x, y in zip(xs, ys))

    def bind(self, ellipsis, sequence: Sequence) -> None:
        """Bind `ellipsis` to `sequence` and merge again the equations that
        wait for it.

        Raises:
            DataConfigMismatchError: If the ellipsis is bound to another
                sequence already that `sequence` contradicts.
        """
        ellipsis = self.representative(ellipsis)
        expanded = self.expand(sequence)
        if expanded == [ellipsis]:
            return
        if ellipsis in expanded:
            raise DataConfigMismatchError(f"{ellipsis} would have to contain itself.")
        if ellipsis in self._bound:
            self.merge_sequences(self._bound[ellipsis], expanded)
            return
        self._bound[ellipsis] = tuple(expanded)
        for equation in self._waiting.pop(ellipsis, ()):
            self.merge_sequences(*equation)
        for dependent in self._dependents.get(ellipsis, ()):
            self._operand_changed(dependent)

    def open_up(self, ellipsis, at_start: bool) -> None:
        """Bind `ellipsis` to one new unknown and a new ellipsis for the rest,
        the unknown at the start or the end.

        Raises:
            StructureUnknown: Unless :attr:`refining`.
        """
        if not self.refining:
            raise StructureUnknown("The structure behind an ellipsis is not known yet.")
        unit, rest = ellipsis.new_unknown(), self._stand_in(ellipsis, {})
        self.bind(ellipsis, (unit, rest) if at_start else (rest, unit))

    def _merge_while_plain(self, xs: list, ys: list) -> int:
        """Merge items from the start while neither is an ellipsis; returns
        how many were merged."""
        count = 0
        for x, y in zip(xs, ys):
            if x.is_ellipsis or y.is_ellipsis:
                break
            self.merge(x, y)
            count += 1
        return count

    def _merge_middles(self, xs: list, ys: list, equation: tuple) -> None:
        """Merge what is left between the merged start and end."""
        if not xs and not ys:
            return
        for single, other in ((xs, ys), (ys, xs)):
            if len(single) == 1 and single[0].is_ellipsis:
                self.bind(single[0], other)
                return
        if not any(item.is_ellipsis for item in xs + ys):
            raise DataConfigMismatchError(
                f"[{', '.join(map(str, xs))}] and [{', '.join(map(str, ys))}] do not match."
            )
        candidates = _alignments(xs, ys)
        alignments = (
            None
            if candidates is None
            else _preferred([a for a in candidates if self._consistent(a)])
        )
        if alignments == []:
            raise DataConfigMismatchError(
                f"[{', '.join(map(str, xs))}] and [{', '.join(map(str, ys))}] can not be aligned."
            )
        if alignments is not None and len(alignments) == 1:
            self._apply(alignments[0])
            return
        for item in xs + ys:
            if item.is_ellipsis:
                self._waiting[item] = self._waiting.get(item, ()) + (equation,)

    def _consistent(self, actions: list) -> bool:
        return all(
            self.compatible(first, second)
            for kind, first, second in actions
            if kind == "merge"
        )

    def _apply(self, actions: list) -> None:
        for kind, first, second in actions:
            if kind == "merge":
                self.merge(first, second)
            else:
                self.bind(first, second)

    # endregion
    # region: Reading

    def resolve(self, config):
        """`config` as known so far: every object replaced by the
        representative of its group, every bound ellipsis by what it stands
        for. Where the representative does not show what is known (other
        dims of an axes, a size only this matching found), a new object is
        built once per group, so equal axes stay one object; an operation dim
        is built over its resolved operands, so it follows HyperParameters
        (e.g. a changed batch size).

        Returns:
            DynamicDataConfiguration: A configuration of this matching.
        """
        return DynamicDataConfiguration(
            *(self._resolved(axes) for axes in self.expand(config.axes)),
            matching=self,
            dtype=config.dtype,
            element_dtype=config.element_dtype,
        )

    def _resolved(self, item):
        """`item` (a representative) as known to this matching."""
        parts = tuple(self._resolved(part) for part in self.expand(item.parts))
        if parts != tuple(item.parts):
            item = self._joined((item, parts), lambda: item.with_parts(parts), item)
        size = self.size(item)
        if size is None or item.size == size:
            return item
        return self._joined(item, lambda: AxesDim(size, item.broadcastable), item)

    def _joined(self, key, build, member):
        """The object built for `key`, once; it joins the group of
        `member`."""
        if key not in self._built:
            new = build()
            self._parent[new] = self.representative(member)
            self._entered.add(new)
            self._built[key] = new
        return self._built[key]

    # endregion


#: How the matching merges each kind of item, checked in order (an ellipsis
#: of axes is an axes).
_MERGE = ((Axes, Matching._merge_axes), (AxesDim, Matching._merge_dims))

#: How the matching tells whether two items of each kind could be merged.
_COMPATIBLE = ((Axes, Matching._axes_compatible), (AxesDim, Matching._dims_compatible))


def _for_kind(table: tuple, item):
    return next(function for cls, function in table if isinstance(item, cls))


def _alignments(xs: list, ys: list) -> list | None:
    """Candidate alignments of two middles, each a list of actions; None if
    there are too many kinds of them to tell."""
    if not any(y.is_ellipsis for y in ys):
        return list(_spread(xs, ys))
    if not any(x.is_ellipsis for x in xs):
        return list(_spread(ys, xs))
    for front, back in ((xs, ys), (ys, xs)):
        if _ends_open(front) and _starts_open(back):
            return _overlaps(front, back)
    return None


def _spread(pattern: list, items: list):
    """Every way to spread `items` over `pattern`, whose ellipses take any
    number of items and whose other entries take one each."""
    if not pattern:
        if not items:
            yield []
        return
    head, rest = pattern[0], pattern[1:]
    if not head.is_ellipsis:
        if items:
            for actions in _spread(rest, items[1:]):
                yield [("merge", head, items[0])] + actions
        return
    for taken in range(len(items) + 1):
        for actions in _spread(rest, items[taken:]):
            yield [("bind", head, items[:taken])] + actions


def _ends_open(items: list) -> bool:
    return items[-1].is_ellipsis and not any(i.is_ellipsis for i in items[:-1])


def _starts_open(items: list) -> bool:
    return items[0].is_ellipsis and not any(i.is_ellipsis for i in items[1:])


def _overlaps(front: list, back: list) -> list:
    """Alignments of `[x1 .. xn, E]` with `[F, y1 .. ym]`: the general one,
    in which a new ellipsis `G` lies between (`F = [x1 .. xn, G]`,
    `E = [G, y1 .. ym]`), and those in which the `x` overlap the `y`."""
    xs, end = front[:-1], front[-1]
    start, ys = back[0], back[1:]
    between = type(end)()
    alignments = [[("bind", start, xs + [between]), ("bind", end, [between] + ys)]]
    for overlap in range(1, min(len(xs), len(ys)) + 1):
        merges = [("merge", x, y) for x, y in zip(xs[len(xs) - overlap :], ys[:overlap])]
        binds = [("bind", start, xs[: len(xs) - overlap]), ("bind", end, ys[overlap:])]
        alignments.append(merges + binds)
    return alignments


def _preferred(alignments: list) -> list:
    """The alignments that merge the most: those without a new ellipsis
    between the two sides, if there are any. Where the two sides meet in
    axes that can be one, they are taken to be one."""
    merging = [a for a in alignments if not _introduces_ellipsis(a)]
    return merging or alignments


def _introduces_ellipsis(actions: list) -> bool:
    bound = [item for kind, _, sequence in actions if kind == "bind" for item in sequence]
    return any(item.is_ellipsis for item in bound)
