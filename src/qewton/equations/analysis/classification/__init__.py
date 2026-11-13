"""Classification: what kind of problem a task is.

`rows.py` gives each constraint its region and orders; `equations.py`,
`conditions.py` and `spaces.py` classify equations, conditions and unknowns, and
`classify.py` collects their facts (`facts.py`) into a `Classification`.
"""

from qewton.equations.analysis.classification.classify import Classification
from qewton.equations.analysis.classification.conditions import ConditionKind
from qewton.equations.analysis.classification.equations import PdeType
from qewton.equations.analysis.classification.facts import Fact
from qewton.equations.analysis.classification.spaces import FunctionSpace

__all__ = ["Classification", "ConditionKind", "PdeType", "Fact", "FunctionSpace"]
