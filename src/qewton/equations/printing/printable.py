"""Mixin that gives symbolic objects a readable str() and notebook display.

The printer lives in pretty.py, which sets `printer` on import. This module imports
nothing, so the symbolic classes can use it without an import cycle.
"""

from __future__ import annotations

from collections.abc import Callable


class Printable:
    printer: Callable[[object], str] | None = (
        None  # set by qewton.equations.printing.pretty
    )

    def __str__(self) -> str:
        printer = self.printer
        return object.__str__(self) if printer is None else printer(self)

    def _ipython_display_(self):
        # IPython's dataclass printer would otherwise win over _repr_pretty_.
        from IPython.display import display  # only called inside IPython

        display({"text/plain": str(self)}, raw=True)
