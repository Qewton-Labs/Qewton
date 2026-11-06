"""Texts as Matplotlib displays them.

Matplotlib's mathtext renders the core's texts as they are, plain text
mixed with ``$...$`` math. It knows a subset of LaTeX: a text whose math it
cannot parse is shown as plain text, with a warning.
"""

import functools
import warnings

from matplotlib.mathtext import MathTextParser

from qewton.visualization.renderers.text import as_plain, segments


def as_mathtext(text: str | None) -> str | None:
    """`text` for mathtext: unchanged if mathtext parses it, else as plain
    text, with a warning.

    Args:
        text: A text, possibly with ``$...$`` parts.

    Returns:
        str or None: The text to display.

    Warns:
        UserWarning: If mathtext cannot parse the math of `text`.
    """
    if not text or not any(is_math for _, is_math in segments(text)):
        return text
    if _parses(text):
        return text
    warnings.warn(
        f"Matplotlib's mathtext cannot render {text!r}; it is shown as plain text.",
        stacklevel=2,
    )
    return as_plain(text)


@functools.cache
def _parses(text: str) -> bool:
    try:
        MathTextParser("path").parse(text)
    except ValueError:
        return False
    return True
