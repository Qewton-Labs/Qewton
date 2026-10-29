"""The task: what to do with a problem.

A Problem is mathematics only. A Task says which functions are unknown, which are
given (once, or once per instance of a batch), and supplies their values. One
problem can carry many tasks: a forward solve, a parameter sweep, an inverse problem.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from qewton.config.variables import Variable
from qewton.equations.analysis.axes import free_axes
from qewton.equations.analysis.diagnostics import Diagnostic, warning
from qewton.equations.analysis.shapes import UnknownShapeError, shape_of
from qewton.equations.expr.nodes import Expr, FunctionSymbol
from qewton.equations.geometry.coordinates import root_of
from qewton.equations.geometry.sets import Product, SetNode

if TYPE_CHECKING:
    from qewton.equations.model.problem import Problem


class TaskError(ValueError):
    """The roles or values of a task do not fit its problem."""


class Role(Enum):
    UNKNOWN = "unknown"  # solve for this
    FIXED = "fixed"  # one known value
    VARYING = "varying"  # known per instance; adds a parameter (batch) axis


@dataclass(frozen=True, eq=False)
class Task:
    """A problem with roles. Built by `problem.task(...)`.

    `roles` covers every function of the problem and every set that was given a value.
    `values` holds what the user supplied: expressions, numbers, arrays or meshes.
    `batch_axes` maps each varying function or set to the parameter axis created for it.

    Sets are fixed unless said otherwise. A set is only listed when it needs a value,
    namely a mesh for arrays that live on it. If the set itself changes from instance
    to instance (a moving boundary), list it as varying, with a mesh per instance. Sets
    derived from a varying set (its boundary, its parts, products with it) vary too,
    and so do the other parts of a partition when one part varies; `varies(set)` tells.
    """

    problem: Problem
    roles: Mapping[FunctionSymbol | SetNode, Role]
    values: Mapping[FunctionSymbol | SetNode, object]
    batch_axes: Mapping[FunctionSymbol | SetNode, Variable]
    diagnostics: tuple[Diagnostic, ...] = field(default=())

    def with_role(self, role: Role) -> tuple[FunctionSymbol, ...]:
        """The functions that have this role, in order of name."""
        return tuple(
            sorted(
                (s for s, r in self.roles.items() if r is role and _is_symbol(s)),
                key=lambda s: s.variable.name,
            )
        )

    def varies(self, node: SetNode) -> bool:
        """Whether the set changes per instance.

        It does if it is varying, if a set it is built from varies (a boundary of a
        varying domain, a factor of a product), or if it is a part of a partition
        whose other parts vary: the parts fill the whole, so one cannot move alone.
        A fixed whole with varying parts stays fixed.
        """
        if self.roles.get(node) is Role.VARYING:
            return True
        if isinstance(node, Product):
            built_from = node.factors
        else:
            built_from = (node.of,) if hasattr(node, "of") else ()
        if any(self.varies(p) for p in built_from):
            return True
        return any(
            self.roles.get(sibling) is Role.VARYING
            for partition in self.problem.partitions
            if node in partition.parts
            for sibling in partition.parts
        )

    @property
    def unknowns(self) -> tuple[FunctionSymbol, ...]:
        return self.with_role(Role.UNKNOWN)

    def __str__(self) -> str:
        lines = []
        for role in Role:
            names = ", ".join(s.variable.name for s in self.with_role(role))
            if names:
                lines.append(f"{role.value}: {names}")
        sets = [
            str(k)
            for k, r in self.roles.items()
            if r is Role.VARYING and not _is_symbol(k)
        ]
        if sets:
            lines.append(f"varying sets: {', '.join(sorted(sets))}")
        for owner, axis in self.batch_axes.items():
            lines.append(f"batch axis of {_name(owner)}: {axis.name}")
        return "\n".join(lines)


def _is_symbol(key: object) -> bool:
    return isinstance(key, FunctionSymbol)


def _as_mapping(entries: object, what: str) -> dict:
    """Accept {symbol: value} or a plain list of symbols (without values)."""
    if entries is None:
        return {}
    if isinstance(entries, Mapping):
        return dict(entries)
    if isinstance(entries, (str, bytes)) or not hasattr(entries, "__iter__"):
        raise TypeError(f"`{what}` must be a list of symbols or a dict of values.")
    return {key: None for key in entries}


def _with_factors(sets) -> set[SetNode]:
    """The sets and, recursively, the factors of their products (a mesh of Omega
    serves Omega x I)."""
    found: set[SetNode] = set()
    for node in sets:
        found.add(node)
        if isinstance(node, Product):
            found |= _with_factors(node.factors)
    return found


def _check_key(problem: Problem, key: object, what: str) -> None:
    if isinstance(key, FunctionSymbol):
        if key not in problem.symbols():
            raise TaskError(f"`{what}`: {key.variable.name} is not in the problem.")
    elif isinstance(key, SetNode):
        if key not in _with_factors(problem.sets()):
            raise TaskError(f"`{what}`: the set {key} is not in the problem.")
    else:
        raise TypeError(
            f"`{what}` needs functions or sets as keys, got {type(key).__name__}."
        )


def _declared_shape(symbol: FunctionSymbol) -> tuple[int, ...] | None:
    try:
        return shape_of(symbol)
    except UnknownShapeError:
        return None


def _check_value(
    problem: Problem,
    symbol: FunctionSymbol,
    value: object,
    meshes: frozenset[Variable],
    varying: bool,
) -> None:
    name = symbol.variable.name
    allowed = free_axes(symbol)
    if isinstance(value, Expr):
        extra = free_axes(value) - allowed
        if extra:
            axes = ", ".join(sorted(a.name for a in extra))
            raise TaskError(
                f"The value of {name} depends on {axes}, which is not among its inputs."
            )
        return
    array = np.asarray(value)
    if not allowed:
        expected = _declared_shape(symbol)
        got = array.shape[1:] if varying else array.shape
        if expected is not None and got != expected:
            raise TaskError(
                f"The value of {name} has shape {got}, but {name} is declared "
                f"with shape {expected}."
            )
        return
    needed = {root_of(a) for a in allowed}
    if problem.time is not None:
        needed.discard(root_of(problem.time))
    missing = needed - meshes
    if missing:
        axes = ", ".join(sorted(a.name for a in missing))
        raise TaskError(
            f"The value of {name} is an array over {axes}; give a mesh for it by "
            "passing the set it lives on, e.g. fixed={Omega: mesh}."
        )


def _batch_axis(problem: Problem, key: object, taken: set[str]) -> Variable:
    name = f"p_{_name(key)}"
    used = {a.name for a in problem.coordinates()} | taken
    while name in used:
        name += "_"
    return Variable(name, dim=1)


def _collect(problem: Problem, entries) -> tuple[dict, dict]:
    """Merge the role lists into `roles` and `values`, checking each entry."""
    roles: dict = {}
    values: dict = {}
    for role, mapping in entries:
        for key, value in mapping.items():
            _check_key(problem, key, role.value)
            if isinstance(key, SetNode) and role is Role.UNKNOWN:
                raise TaskError("A set cannot be unknown; only functions are solved for.")
            if key in roles:
                raise TaskError(
                    f"{_name(key)} is listed as {roles[key].value} and as {role.value}."
                )
            if role is Role.UNKNOWN and value is not None:
                raise TaskError(f"The unknown {_name(key)} cannot be given a value.")
            roles[key] = role
            if value is not None:
                values[key] = value
    return roles, values


def _fill_defaults(problem: Problem, roles: dict, values: dict) -> list[Diagnostic]:
    """Make every unlisted function FIXED; flag those without a value."""
    found = []
    for symbol in sorted(problem.symbols(), key=lambda s: s.variable.name):
        if symbol not in roles:
            roles[symbol] = Role.FIXED
            if symbol not in values:
                found.append(
                    warning(
                        f"{symbol.variable.name} is neither unknown nor given a value; "
                        "it stays a symbolic fixed function.",
                        symbol,
                    )
                )
    return found


def build_task(
    problem: Problem,
    unknown=(),
    fixed=None,
    varying=None,
) -> Task:
    """Assign roles. Every function not listed is FIXED, and flagged if it has no value.

    Raises TaskError for a listed function that is not in the problem, for a function
    with two roles, for an unknown with a value, and for a value that does not fit
    the function's declared inputs or shape.
    """
    roles, values = _collect(
        problem,
        [
            (Role.UNKNOWN, _as_mapping(unknown, "unknown")),
            (Role.FIXED, _as_mapping(fixed, "fixed")),
            (Role.VARYING, _as_mapping(varying, "varying")),
        ],
    )
    meshes = frozenset(
        root_of(axis) for key in values if isinstance(key, SetNode) for axis in key.axes
    )
    for key, value in values.items():
        if isinstance(key, FunctionSymbol):
            _check_value(problem, key, value, meshes, roles[key] is Role.VARYING)
    diagnostics = _fill_defaults(problem, roles, values)

    axes: dict[FunctionSymbol | SetNode, Variable] = {}
    varying_keys = sorted((k for k, r in roles.items() if r is Role.VARYING), key=_name)
    for key in varying_keys:
        axes[key] = _batch_axis(problem, key, {a.name for a in axes.values()})

    return Task(problem, roles, values, axes, tuple(diagnostics))


def _name(key: object) -> str:
    return key.variable.name if isinstance(key, FunctionSymbol) else str(key)
