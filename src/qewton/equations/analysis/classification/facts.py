"""Fact: one classification result together with the rule and the reason behind it."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Fact:
    """A finding of `classify`.

    `rule` is a stable id such as "pde_type", `subject` is what the fact is about (a
    constraint, an unknown, a set), `value` the finding, `why` a sentence saying how it
    was derived, and `constraints` the zero-based indices of the (normalized)
    constraints it was read from.
    """

    rule: str
    subject: object
    value: object
    why: str
    constraints: tuple[int, ...] = ()

    def __str__(self) -> str:
        value = getattr(self.value, "value", self.value)
        return f"{self.subject}: {value}  ({self.why})"


def by_constraint(facts: Iterable[Fact]) -> dict[int, Any]:
    """The value of each fact by the index of the constraint it is about.

    For facts about one constraint each, such as the PDE type of an equation or the
    kind of a condition.
    """
    return {fact.constraints[0]: fact.value for fact in facts}
