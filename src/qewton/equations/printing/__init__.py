"""Presentation: readable text (`pretty`, in `text.py`) and the coupling diagram.

It sits on top of the other layers: `pretty` prints the model, and the coupling
diagram (`diagram.py`) draws analysis results. The `Printable` mixin that the
symbolic classes inherit lives in `equations/printable.py`, so they do not depend
on this package.
"""

from qewton.equations.printing.diagram import (
    CouplingDiagram,
    coupling_diagram,
    plot_coupling,
)
from qewton.equations.printing.text import pretty

__all__ = ["CouplingDiagram", "coupling_diagram", "plot_coupling", "pretty"]
