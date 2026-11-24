"""The color scales of a color channel, shared by every plot family."""

from collections.abc import Callable

from qewton.visualization.axes import Selector
from qewton.visualization.scales import ColorScale


def candidate_scales(
    color,
    color_scale: ColorScale | list[ColorScale] | None,
    new_scale: Callable[[object], ColorScale] = lambda candidate: ColorScale(),
) -> list[ColorScale]:
    """One ColorScale per candidate of a color binding.

    Args:
        color: The color reference, a Selector of references, or None.
        color_scale: One ColorScale shared by every candidate, a list with
            one per candidate, or None for a new one per candidate.
        new_scale: Makes the new scale of a candidate.

    Returns:
        list of ColorScale: One per candidate; empty without a color
            binding.

    Raises:
        ValueError: If `color_scale` is a list that does not give one
            scale per candidate.
    """
    if color is None:
        return []
    candidates = color.candidates if isinstance(color, Selector) else [color]
    if isinstance(color_scale, list):
        if len(color_scale) != len(candidates):
            raise ValueError(
                f"color_scale gives {len(color_scale)} scales for {len(candidates)} "
                "color candidate(s); give one per Selector candidate."
            )
        return list(color_scale)
    if color_scale is None:
        return [new_scale(candidate) for candidate in candidates]
    return [color_scale] * len(candidates)


def unique(scales: list[ColorScale]) -> list[ColorScale]:
    """Each scale once, in order.

    Args:
        scales: Scales, possibly repeated.

    Returns:
        list of ColorScale: The distinct instances.
    """
    found = []
    for scale in scales:
        if not any(scale is other for other in found):
            found.append(scale)
    return found
