"""The model layer: constraints, the Problem that collects them, and Tasks.

A `Problem` is mathematics only; a `Task` says what to do with it: which functions
are unknown, which are given, and which vary per instance.
"""

from qewton.equations.model.constraints import (
    Complementary,
    Constraint,
    Eq,
    Ge,
    Le,
    PendingConstraint,
    Relation,
)
from qewton.equations.model.observation import Observed
from qewton.equations.model.problem import Problem
from qewton.equations.model.task import Role, Task, TaskError

__all__ = [
    "Complementary",
    "Constraint",
    "Eq",
    "Ge",
    "Le",
    "PendingConstraint",
    "Relation",
    "Observed",
    "Problem",
    "Role",
    "Task",
    "TaskError",
]
