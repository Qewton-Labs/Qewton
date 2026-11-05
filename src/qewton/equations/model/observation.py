"""Observed: what data exists for an unknown function.

An unknown stays an unknown, whether or not samples of it exist. `Observed` records
what the task knows about those samples, so the analysis can tell physics-informed
training (no data) from data-driven training (data everywhere) and hybrids.

    problem.task(unknown=[u], observed={u: Observed(on=Omega * I, samples=2000)})
"""

from __future__ import annotations

from dataclasses import dataclass

from qewton.equations.geometry.sets import SetNode


@dataclass(frozen=True)
class Observed:
    """Samples of an unknown function.

    `on` is the set the samples lie on, or None for everywhere the function lives.
    `samples` is how many there are per instance, or None if that is not known yet.
    `noisy` says whether the samples carry measurement noise.
    """

    on: SetNode | None = None
    samples: int | None = None
    noisy: bool = False

    def __post_init__(self):
        if self.on is not None and not isinstance(self.on, SetNode):
            raise TypeError(f"`on` must be a set, got {type(self.on).__name__}.")
        if self.samples is not None and (
            isinstance(self.samples, bool)
            or not isinstance(self.samples, int)
            or self.samples < 1
        ):
            raise ValueError(
                f"`samples` must be a positive integer, got {self.samples!r}."
            )

    def __str__(self) -> str:
        parts = [f"on {self.on}" if self.on is not None else "everywhere"]
        if self.samples is not None:
            parts.append(f"{self.samples} samples")
        if self.noisy:
            parts.append("noisy")
        return ", ".join(parts)
