"""Structure: what the equations look like and how the unknowns couple.

`structure.py` has derivative orders, the operator/source split and linearity,
`coupling.py` the scalar coupling graph, and `batching.py` with `batch_axes.py`
what each parameter axis does to the problem.
"""

from qewton.equations.analysis.structure.batching import (
    BatchEntry,
    BatchMode,
    Batching,
)
from qewton.equations.analysis.structure.coupling import (
    CouplingBlock,
    CouplingGraph,
    EquationNode,
    UnknownNode,
)
from qewton.equations.analysis.structure.structure import (
    BatchPlacement,
    Split,
    Term,
    constraint_orders,
    derivative_orders,
    linearity,
    unknown_orders,
)

__all__ = [
    "BatchEntry",
    "BatchMode",
    "Batching",
    "CouplingBlock",
    "CouplingGraph",
    "EquationNode",
    "UnknownNode",
    "BatchPlacement",
    "Split",
    "Term",
    "constraint_orders",
    "derivative_orders",
    "linearity",
    "unknown_orders",
]
