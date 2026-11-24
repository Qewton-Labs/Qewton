"""Walking and rebuilding expression trees, through `Expr.children()` alone."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator

from qewton.equations.expr.base import Expr
from qewton.equations.expr.leaves import FunctionSymbol


def walk(expr: Expr) -> Iterator[Expr]:
    """Yield `expr` and all its sub-expressions, depth first."""
    yield expr
    for child in expr.children():
        yield from walk(child)


def transform(expr: Expr, rewrite: Callable[[Expr], Expr]) -> Expr:
    """Rebuild `expr` bottom-up, applying `rewrite` to every node after its children.

    A node whose children did not change is reused, so an unchanged tree comes back
    as the same object.
    """
    old = expr.children()
    new = tuple(transform(child, rewrite) for child in old)
    same = all(n is o for n, o in zip(new, old))
    return rewrite(expr if same else expr.with_children(new))


def symbols_in(expr: Expr) -> Iterator[FunctionSymbol]:
    """Yield every function symbol in `expr`, depth first, repeats included."""
    return (node for node in walk(expr) if isinstance(node, FunctionSymbol))


def contains(expr: Expr, symbols: Collection[FunctionSymbol]) -> bool:
    """Whether one of `symbols` occurs in `expr`."""
    return any(symbol in symbols for symbol in symbols_in(expr))
