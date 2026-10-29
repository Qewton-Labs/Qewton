"""A summary of what a problem contains: coordinates, sets and functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qewton.config.variables import Variable
from qewton.equations.analysis.infer import infer_shapes
from qewton.equations.expr.nodes import FunctionSymbol
from qewton.equations.geometry.coordinates import coordinate_factors, root_of
from qewton.equations.geometry.sets import Boundary, Domain, Part, Product, SetNode, Slice

if TYPE_CHECKING:
    from qewton.equations.model.problem import Problem

Shape = tuple[int, ...]


@dataclass(frozen=True)
class Overview:
    """What a problem is made of. `print(overview(problem))` shows it.

    `time` is the declared time coordinate, if any, and `space` the other coordinates.
    `sets` lists every set, including the factors of products. `functions` maps each
    function to its output shape, and `assumed` holds those whose shape was not
    declared and was taken to be scalar.
    """

    time: Variable | None
    space: tuple[Variable, ...]
    sets: tuple[SetNode, ...]
    functions: dict[FunctionSymbol, Shape]
    assumed: frozenset[FunctionSymbol]
    constraints: int

    def __str__(self) -> str:
        lines = ["coordinates:"]
        for axis in self.space:
            lines.append(f"  {axis.name}  space, dim {axis.dim}")
        if self.time is not None:
            lines.append(f"  {self.time.name}  time")
        lines.append("sets:")
        lines += [f"  {node}  {_describe_set(node)}" for node in self.sets]
        lines.append("functions:")
        for symbol, shape in self.functions.items():
            note = " (assumed)" if symbol in self.assumed else ""
            lines.append(f"  {_signature(symbol)}  {_describe_shape(shape)}{note}")
        lines.append(f"constraints: {self.constraints}")
        return "\n".join(lines)


def _signature(symbol: FunctionSymbol) -> str:
    if symbol.inputs is None:
        return f"{symbol.variable.name}  (constant)"
    axes = ", ".join(a.name for a in coordinate_factors(symbol.inputs))
    return f"{symbol.variable.name}({axes})"


def _describe_shape(shape: Shape) -> str:
    if shape == ():
        return "scalar"
    if len(shape) == 1:
        return f"vector({shape[0]})"
    return f"tensor{shape}"


def _describe_set(node: SetNode) -> str:
    kinds = {
        Domain: lambda n: f"domain in {', '.join(sorted(a.name for a in n.axes))}",
        Boundary: lambda n: f"boundary of {n.of}",
        Part: lambda n: f"part of {n.of}",
        Slice: lambda n: f"slice of {n.axis.name}",
        Product: lambda n: "product",
    }
    describe = kinds.get(type(node), lambda n: type(n).__name__)
    return describe(node)


def _with_factors(nodes) -> set[SetNode]:
    found: set[SetNode] = set()
    for node in nodes:
        found.add(node)
        if isinstance(node, Product):
            found |= _with_factors(node.factors)
    return found


def overview(problem: Problem) -> Overview:
    """Collect the coordinates, sets and functions of a problem."""
    inference = infer_shapes(problem)
    time = None if problem.time is None else root_of(problem.time)
    space = tuple(
        sorted((a for a in problem.coordinates() if a != time), key=lambda a: a.name)
    )
    sets = sorted(_with_factors(problem.sets()), key=lambda s: (s.codim, str(s)))
    functions = dict(sorted(inference.shapes.items(), key=lambda kv: str(kv[0])))
    return Overview(
        time,
        space,
        tuple(sets),
        functions,
        frozenset(inference.defaulted),
        len(problem.constraints),
    )
