"""The suggestion mechanism: `Rule`, `Suggestion` and `suggest`.

A rule looks at the analysis of a task and proposes a solution method. Write one by
subclassing `Rule`: set `method` (what is suggested) and `category`, and implement
`check`, which yields one suggestion for each reason the method applies:

    class Newton(Rule):
        category = Category.SOLVER
        method = "Newton (or Picard) iteration"

        def check(self, ctx):
            if ctx.nonlinear:
                yield self.suggestion("an unknown occurs nonlinearly in the operator")

Defining the class registers it, and `suggest` runs every registered rule in the
order the rules were defined. `ctx` is the task's `AnalysisContext`: the task, its
classification, batching, linearity, decoupled axes and training modes. Rules only
read from it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from qewton.equations.analysis.classification.facts import Fact
from qewton.equations.model.task import Task

if TYPE_CHECKING:
    from qewton.equations.analysis.context import AnalysisContext


class Category(Enum):
    """What part of the solution a suggestion is about."""

    DISCRETIZATION = "discretization"
    SOLVER = "solver"
    BATCH = "batch"
    LEARNING = "learning"


@dataclass(frozen=True)
class Suggestion:
    """A method, the rule that proposed it, and why.

    `rule` is the class name of the rule. `evidence` holds the facts the suggestion
    was derived from, so a report can always explain itself.
    """

    rule: str
    method: str
    why: str
    evidence: tuple[Fact, ...] = ()
    category: Category = Category.DISCRETIZATION

    def __str__(self) -> str:
        return f"{self.method}  <- {self.why}"


class Rule:
    """Base class of the suggestion rules.

    A subclass sets `method` and `category` and implements `check`. Defining it
    registers it in `Rule.registry` under its class name. A later class with the same
    name replaces the earlier one, which keeps re-running a notebook cell harmless.

    Pass `register=False` for a class that should not run by itself: a base class
    that subclasses fill in with data, or a rule you only want to pass to `suggest`
    explicitly.

        class BlockTypeRule(Rule, register=False):
            kind: ClassVar[PdeType]
            ...
    """

    method: ClassVar[str]
    category: ClassVar[Category] = Category.DISCRETIZATION
    registry: ClassVar[dict[str, type[Rule]]] = {}

    def __init_subclass__(cls, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if not register:
            return
        if not isinstance(getattr(cls, "method", None), str):
            raise TypeError(f"The rule {cls.__name__} needs a `method`.")
        Rule.registry[cls.__name__] = cls

    def check(self, ctx: AnalysisContext) -> Iterator[Suggestion]:
        """Yield one suggestion for each reason `method` applies to the task."""
        raise NotImplementedError(f"{type(self).__name__} does not implement check.")

    def suggestion(
        self, why: str, evidence: Iterable[Fact] = (), method: str | None = None
    ) -> Suggestion:
        """A suggestion by this rule. `method` replaces the class's method if given."""
        return Suggestion(
            type(self).__name__,
            self.method if method is None else method,
            why,
            tuple(evidence),
            self.category,
        )


def suggest(
    task: Task | AnalysisContext, rules: Iterable[type[Rule]] | None = None
) -> tuple[Suggestion, ...]:
    """The suggestions for `task`, in rule order.

    Runs every registered rule, or only `rules` if given. `task` may also be an
    `AnalysisContext`, whose results are then reused.
    """
    if isinstance(task, Task):
        # The context imports this module, so it can only be imported here.
        from qewton.equations.analysis.context import AnalysisContext

        task = AnalysisContext(task)
    chosen = Rule.registry.values() if rules is None else rules
    return tuple(s for rule in chosen for s in rule().check(task))
