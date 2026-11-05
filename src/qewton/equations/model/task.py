"""The task: what to do with a problem.

A Problem is mathematics only. A Task says which functions are unknown, which are
given (once, or once per instance of a batch), and supplies their values. One
problem can carry many tasks: a forward solve, a parameter sweep,
an inverse problem.
"""

from __future__ import annotations

from collections.abc import Mapping, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from qewton.config.variables import Variable
from qewton.equations.analysis.checks.axes import free_axes
from qewton.equations.analysis.checks.diagnostics import Diagnostic, warning
from qewton.equations.analysis.checks.shapes import UnknownShapeError, shape_of
from qewton.equations.expr.nodes import Expr, FunctionSymbol
from qewton.equations.geometry.coordinates import root_of
from qewton.equations.geometry.sets import Product, SetNode
from qewton.equations.model.observation import Observed

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

    `observed` says which unknowns come with samples. It is not a role: an observed
    function is still an unknown.
    """

    problem: Problem
    roles: Mapping[FunctionSymbol | SetNode, Role]
    values: Mapping[FunctionSymbol | SetNode, object]
    batch_axes: Mapping[FunctionSymbol | SetNode, Variable]
    diagnostics: tuple[Diagnostic, ...] = field(default=())
    observed: Mapping[FunctionSymbol, Observed] = field(default_factory=dict)

    def with_role(self, role: Role) -> tuple[FunctionSymbol, ...]:
        """The functions that have this role, in order of name."""
        return tuple(
            sorted(  # type: ignore
                (s for s, r in self.roles.items() if r is role and _is_symbol(s)),
                key=lambda s: (
                    s.variable.name
                    if isinstance(s, FunctionSymbol) and s.variable.name is not None
                    else ""
                ),
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
            built_from = (node.of,) if hasattr(node, "of") else ()  # type: ignore
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
            names = ", ".join(str(s.variable.name) for s in self.with_role(role))
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
        for symbol, data in sorted(self.observed.items(), key=lambda e: _name(e[0])):
            lines.append(f"observed {_name(symbol)}: {data}")
        return "\n".join(lines)


def _is_symbol(key: object) -> bool:
    return isinstance(key, FunctionSymbol)


def _as_mapping(entries: object, what: str) -> dict:
    """Accept {symbol: value} or a plain list of symbols (without values)."""
    if entries is None:
        return {}
    if isinstance(entries, Mapping):
        return dict(entries)
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Iterable):
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
    """Check that `value` fits the function `symbol`, and raise a TaskError if not.

    An expression must only use the function's own input coordinates. A number or an
    array must have the declared output shape. If it is sampled over the function's
    inputs, each of those coordinates also needs a mesh, which is given by passing the
    set it lives on: fixed={Omega: mesh}. `meshes` holds the coordinates that have one,
    and `varying` says the first axis of an array is the batch axis, one entry per
    instance. Array lengths are not compared with the meshes yet.
    """
    if isinstance(value, Expr):
        _check_expression(symbol, value)
    else:
        _check_array(problem, symbol, np.asarray(value), meshes, varying)


def _check_expression(symbol: FunctionSymbol, value: Expr) -> None:
    """An expression carries no shape here: it only has to stay within the inputs."""
    extra = free_axes(value) - free_axes(symbol)
    if extra:
        axes = ", ".join(sorted(str(a.name) for a in extra))
        raise TaskError(
            f"The value of {symbol.variable.name} depends on {axes}, "
            "which is not among its inputs."
        )


def _check_array(
    problem: Problem,
    symbol: FunctionSymbol,
    array: np.ndarray,
    meshes: frozenset[Variable],
    varying: bool,
) -> None:
    name = symbol.variable.name
    inputs = free_axes(symbol)
    expected = _declared_shape(symbol)
    # Drop the batch axis of a varying function: what is left is one instance.
    got = array.shape[1:] if varying else array.shape

    if not inputs:
        # A function without inputs is a constant, so only its shape can be wrong.
        if expected is not None and got != expected:
            raise TaskError(
                f"The value of {name} has shape {got}, but {name} is declared "
                f"with shape {expected}."
            )
        return
    if got == (expected or ()):
        return  # one value of the function's shape: a constant, no mesh needed

    # Otherwise the array is sampled over the inputs, and each of them needs a mesh.
    missing = _meshed_inputs(problem, inputs) - meshes
    if missing:
        axes = ", ".join(sorted(str(a.name) for a in missing))
        raise TaskError(
            f"The value of {name} is an array over {axes}; give a mesh for it by "
            "passing the set it lives on, e.g. fixed={Omega: mesh}."
        )


def _meshed_inputs(problem: Problem, inputs) -> set[Variable]:
    """The inputs an array has to be sampled over. Time is not meshed."""
    needed = {root_of(axis) for axis in inputs}
    if problem.time is not None:
        needed.discard(root_of(problem.time))
    return needed


def _batch_axis(problem: Problem, key: object, taken: set[str]) -> Variable:
    name = f"p_{_name(key)}"
    used = {a.name for a in problem.coordinates()} | taken
    while name in used:
        name += "_"
    return Variable(name, dim=1)


def _check_entry(problem: Problem, key: object, value: object, role: Role, roles: dict):
    """Check one listed function or set before it is given its role."""
    _check_key(problem, key, role.value)
    if isinstance(key, SetNode) and role is Role.UNKNOWN:
        raise TaskError("A set cannot be unknown; only functions are solved for.")
    if key in roles:
        raise TaskError(
            f"{_name(key)} is listed as {roles[key].value} and as {role.value}."
        )
    if role is Role.UNKNOWN and value is not None:
        raise TaskError(f"The unknown {_name(key)} cannot be given a value.")


def _collect(problem: Problem, entries) -> tuple[dict, dict]:
    """Merge the role lists into `roles` and `values`, checking each entry."""
    roles: dict = {}
    values: dict = {}
    for role, mapping in entries:
        for key, value in mapping.items():
            _check_entry(problem, key, value, role, roles)
            roles[key] = role
            if value is not None:
                values[key] = value
    return roles, values


def _fill_defaults(problem: Problem, roles: dict, values: dict) -> list[Diagnostic]:
    """Make every unlisted function FIXED; flag those without a value."""
    unlisted = [
        symbol
        for symbol in sorted(problem.symbols(), key=lambda s: str(s.variable.name))
        if symbol not in roles
    ]
    for symbol in unlisted:
        roles[symbol] = Role.FIXED
    return [
        warning(
            f"{symbol.variable.name} is neither unknown nor given a value; "
            "it stays a symbolic fixed function.",
            symbol,
        )
        for symbol in unlisted
        if symbol not in values
    ]


def _make_batch_axes(problem: Problem, roles: dict) -> dict:
    """One parameter axis p_<name> per varying function or set, in order of name."""
    axes: dict = {}
    for key in sorted((k for k, r in roles.items() if r is Role.VARYING), key=_name):
        axes[key] = _batch_axis(problem, key, {str(a.name) for a in axes.values()})
    return axes


def build_task(
    problem: Problem,
    unknown=(),
    fixed=None,
    varying=None,
    observed=None,
) -> Task:
    """Assign roles. Every function not listed is FIXED, and flagged if it has no value.

    Raises TaskError for a listed function that is not in the problem, for a function
    with two roles, for an unknown with a value, and for a value that does not fit
    the function's declared inputs or shape. `observed` maps unknowns to `Observed`;
    anything else, or a set outside the function's inputs, raises a TaskError.
    """
    roles, values = _collect(
        problem,
        [
            (Role.UNKNOWN, _as_mapping(unknown, Role.UNKNOWN.value)),
            (Role.FIXED, _as_mapping(fixed, Role.FIXED.value)),
            (Role.VARYING, _as_mapping(varying, Role.VARYING.value)),
        ],
    )
    # A value given for a set is its mesh: it covers the set's coordinates.
    meshes = frozenset(
        root_of(axis) for key in values if isinstance(key, SetNode) for axis in key.axes
    )
    for key, value in values.items():
        if isinstance(key, FunctionSymbol):
            _check_value(problem, key, value, meshes, roles[key] is Role.VARYING)

    diagnostics = _fill_defaults(problem, roles, values)
    axes = _make_batch_axes(problem, roles)
    data = _check_observed(problem, roles, observed)
    return Task(problem, roles, values, axes, tuple(diagnostics), data)


def _check_observed(problem: Problem, roles: dict, observed) -> dict:
    if observed is None:
        return {}
    if not isinstance(observed, Mapping):
        raise TypeError("`observed` must be a dict from unknown functions to Observed.")
    for symbol, data in observed.items():
        _check_key(problem, symbol, "observed")
        if not isinstance(symbol, FunctionSymbol):
            raise TypeError("`observed` needs functions as keys.")
        if roles.get(symbol) is not Role.UNKNOWN:
            raise TaskError(
                f"`observed`: {symbol.variable.name} is not an unknown; "
                "only unknowns have samples to learn from."
            )
        if not isinstance(data, Observed):
            raise TypeError(
                f"`observed` needs Observed values, got {type(data).__name__}."
            )
        if data.on is not None:
            outside = {root_of(a) for a in data.on.axes} - free_axes(symbol)
            if outside:
                names = ", ".join(sorted(str(a.name) for a in outside))
                raise TaskError(
                    f"The samples of {symbol.variable.name} lie on {data.on}, which "
                    f"uses {names}, not among the function's inputs."
                )
    return dict(observed)


def _name(key: object) -> str:
    return str(key.variable.name) if isinstance(key, FunctionSymbol) else str(key)
