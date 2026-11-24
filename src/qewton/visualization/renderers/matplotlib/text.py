"""Texts as Matplotlib displays them.

Matplotlib's mathtext renders the core's texts as they are, plain text
mixed with ``$...$`` math. It knows a subset of LaTeX: a text whose math it
cannot parse is shown as plain text, with a warning. Plain text falls back
to Matplotlib's Computer Modern (``cmr10``), which holds other glyphs at a
few ASCII characters (TeX's encoding, e.g. a dot accent at ``_``); these
are written as math.
"""

import functools
import warnings

from matplotlib.mathtext import MathTextParser

from qewton.visualization.renderers.text import as_plain, segments

#: The characters ``cmr10`` holds other glyphs at, as math.
_AS_MATH = {
    "_": r"$\_$",
    "<": "$<$",
    ">": "$>$",
    "|": "$|$",
    "{": r"$\{$",
    "}": r"$\}$",
    "\\": r"$\backslash$",
}


def as_mathtext(text: str | None) -> str | None:
    """`text` for mathtext: its math unchanged if mathtext parses it, else
    the whole text as plain text, with a warning; in plain parts the
    characters ``cmr10`` lacks are written as math.

    Args:
        text: A text, possibly with ``$...$`` parts.

    Returns:
        str or None: The text to display.

    Warns:
        UserWarning: If mathtext cannot parse the math of `text`.
    """
    if not text:
        return text
    pieces = segments(text)
    if any(is_math for _, is_math in pieces) and not _parses(text):
        warnings.warn(
            f"Matplotlib's mathtext cannot render {text!r}; it is shown as plain text.",
            stacklevel=2,
        )
        pieces = [(as_plain(text), False)]
    return "".join(
        f"${piece}$" if is_math else "".join(_AS_MATH.get(c, c) for c in piece)
        for piece, is_math in pieces
    )


@functools.cache
def _parses(text: str) -> bool:
    try:
        MathTextParser("path").parse(text)
    except ValueError:
        return False
    return True
