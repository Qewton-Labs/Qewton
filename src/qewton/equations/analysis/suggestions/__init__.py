"""suggest: solution methods for a task, each with the evidence behind it.

    for suggestion in suggest(task):
        print(suggestion)

Each suggestion names the method, says why, and carries the facts it was derived
from. The suggestions are starting points, not a ranking: several can apply at once,
for instance a method for the equation type and one for the batching.

`base.py` holds the mechanism (`Rule`, `Suggestion`, `suggest`) and `rules.py` the
built-in rules. Adding a method means adding one `Rule` subclass.
"""

from qewton.equations.analysis.suggestions import (
    rules,
)  # defining the rules registers them
from qewton.equations.analysis.suggestions.base import Category, Rule, Suggestion, suggest

__all__ = ["Category", "Rule", "Suggestion", "suggest"]
