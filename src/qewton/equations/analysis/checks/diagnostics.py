"""Findings of the analysis: errors, warnings and notes.

Errors mean the problem is wrong as written. Warnings mean it is probably not what
was intended, or that the analysis had to guess. Infos report what was done for the
user, e.g. a time lifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Diagnostic:
    """One finding. `where` is the constraint, set or function it is about, if any."""

    severity: Severity
    message: str
    where: object | None = None

    def __str__(self) -> str:
        text = f"{self.severity.value}: {self.message}"
        return text if self.where is None else f"{text}  [{self.where}]"


def errors(diagnostics: tuple[Diagnostic, ...]) -> tuple[Diagnostic, ...]:
    return tuple(d for d in diagnostics if d.severity is Severity.ERROR)


def error(message: str, where: object | None = None) -> Diagnostic:
    return Diagnostic(Severity.ERROR, message, where)


def warning(message: str, where: object | None = None) -> Diagnostic:
    return Diagnostic(Severity.WARNING, message, where)


def info(message: str, where: object | None = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, message, where)
