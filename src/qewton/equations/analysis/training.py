"""training: how an unknown can be learned, from the data the task has for it.

    physics informed   no samples: only the equations give a loss
    data driven        samples everywhere the function lives: supervised learning
    hybrid             samples on part of it: the equations fill the rest

An unknown is covered when its samples lie on, or inside, each of its full-dimensional
sets. Data in the bulk implies data on its boundary and initial slice, so lower
dimensional sets do not count against it.
"""

from __future__ import annotations

from enum import Enum

from qewton.equations.analysis.prepare.signatures import Signatures
from qewton.equations.expr.nodes import FunctionSymbol
from qewton.equations.geometry.sets import is_subset
from qewton.equations.model.task import Task


class TrainingMode(Enum):
    PHYSICS_INFORMED = "physics informed"
    DATA_DRIVEN = "data driven"
    HYBRID = "hybrid"


def _covered(task: Task, table: Signatures, symbol: FunctionSymbol) -> bool:
    data = task.observed[symbol]
    if data.on is None:
        return True
    sets = table[symbol].evaluated_on
    lowest = min((s.codim for s in sets), default=0)
    return all(is_subset(s, data.on) for s in sets if s.codim == lowest)


def build_training_modes(
    task: Task, table: Signatures
) -> dict[FunctionSymbol, TrainingMode]:
    """The training mode of every unknown, in order of name.

    `table` holds the signatures, which say where each unknown is evaluated.
    """
    modes: dict[FunctionSymbol, TrainingMode] = {}
    for symbol in task.unknowns:
        if symbol not in task.observed:
            modes[symbol] = TrainingMode.PHYSICS_INFORMED
        elif _covered(task, table, symbol):
            modes[symbol] = TrainingMode.DATA_DRIVEN
        else:
            modes[symbol] = TrainingMode.HYBRID
    return modes
