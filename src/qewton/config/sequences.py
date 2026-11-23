"""Merging of sequences that may contain ellipses: the axes of a
configuration and the dims of an axes.

An item of a sequence is an axes or a dim; an *ellipsis* (`EllipsisAxes`,
`EllipsisDim`) stands for a sequence of items, possibly empty. Merging two
sequences merges corresponding items and binds ellipses to the parts they
stand for. Where several alignments are consistent, the merge waits: every
ellipsis involved keeps the equation and merges it again once it is bound.
Of the consistent alignments, those that merge the most are preferred (no
new ellipsis between the sides); only a single remaining one is applied, so
the result does not depend on the order of merges.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qewton.config.errors import DataConfigMismatchError


def expand(items: Sequence) -> list:
    """The representatives of `items`, with every bound ellipsis replaced by
    the sequence it is bound to."""
    expanded = []
    for item in items:
        item = item.representative
        if item.is_ellipsis and item.bound_to is not None:
            expanded.extend(expand(item.bound_to))
        else:
            expanded.append(item)
    return expanded


def merge_sequences(first: Sequence, second: Sequence) -> None:
    """Merge two sequences item by item; ellipses take the rest.

    Raises:
        DataConfigMismatchError: If no alignment of the sequences is
            consistent.
    """
    xs, ys = expand(first), expand(second)
    start = _merge_while_plain(xs, ys)
    end = _merge_while_plain(xs[start:][::-1], ys[start:][::-1])
    middle_x, middle_y = xs[start : len(xs) - end], ys[start : len(ys) - end]
    _merge_middles(middle_x, middle_y, (tuple(first), tuple(second)))


def sequences_compatible(first: Sequence, second: Sequence) -> bool:
    """Whether two sequences could be merged; with an ellipsis, only their
    items outside of it are checked."""
    xs, ys = expand(first), expand(second)
    if any(item.is_ellipsis for item in xs + ys):
        return True
    return len(xs) == len(ys) and all(x.compatible(y) for x, y in zip(xs, ys))


def bind(ellipsis: Any, sequence: Sequence) -> None:
    """Bind `ellipsis` to `sequence` and merge again the equations it waits
    for."""
    ellipsis = ellipsis.representative
    expanded = expand(sequence)
    if expanded == [ellipsis]:
        return
    if ellipsis in expanded:
        raise DataConfigMismatchError(f"{ellipsis} would have to contain itself.")
    if ellipsis.bound_to is not None:
        merge_sequences(ellipsis.bound_to, expanded)
        return
    ellipsis.bound_to = tuple(expanded)
    equations, ellipsis.waiting = ellipsis.waiting, ()
    for equation in equations:
        merge_sequences(*equation)
    for dependent in ellipsis._dependents:
        dependent.operand_changed()


def _merge_while_plain(xs: list, ys: list) -> int:
    """Merge items from the start while neither is an ellipsis; returns how
    many were merged."""
    count = 0
    for x, y in zip(xs, ys):
        if x.is_ellipsis or y.is_ellipsis:
            break
        x.merge(y)
        count += 1
    return count


def _merge_middles(xs: list, ys: list, equation: tuple) -> None:
    """Merge what is left between the merged start and end."""
    if not xs and not ys:
        return
    for single, other in ((xs, ys), (ys, xs)):
        if len(single) == 1 and single[0].is_ellipsis:
            bind(single[0], other)
            return
    if not any(item.is_ellipsis for item in xs + ys):
        raise DataConfigMismatchError(
            f"[{', '.join(map(str, xs))}] and [{', '.join(map(str, ys))}] do not match."
        )
    candidates = _alignments(xs, ys)
    alignments = None if candidates is None else _preferred([a for a in candidates if _consistent(a)])
    if alignments == []:
        raise DataConfigMismatchError(
            f"[{', '.join(map(str, xs))}] and [{', '.join(map(str, ys))}] can not be aligned."
        )
    if alignments is not None and len(alignments) == 1:
        _apply(alignments[0])
        return
    for item in xs + ys:
        if item.is_ellipsis:
            item.waiting = item.waiting + (equation,)


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


def _consistent(actions: list) -> bool:
    return all(
        first.compatible(second) for kind, first, second in actions if kind == "merge"
    )


def _apply(actions: list) -> None:
    for kind, first, second in actions:
        if kind == "merge":
            first.merge(second)
        else:
            bind(first, second)
