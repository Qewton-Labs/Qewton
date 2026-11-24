"""Texts of the core as renderers display them.

The core writes names of Variables in math mode (``$x_1$``), everything
else as plain text, and joins them freely, e.g. ``"$t$ = 0.5"``. A text
with an odd number of ``$`` is plain as a whole. :func:`segments` splits a
text into its plain and math parts; :func:`as_plain` turns it into plain
text, for displays without math.
"""

import re

GREEK = {
    name: chr(code)
    for name, code in (
        ("alpha", 0x3B1), ("beta", 0x3B2), ("gamma", 0x3B3), ("delta", 0x3B4),
        ("epsilon", 0x3B5), ("zeta", 0x3B6), ("eta", 0x3B7), ("theta", 0x3B8),
        ("iota", 0x3B9), ("kappa", 0x3BA), ("lambda", 0x3BB), ("mu", 0x3BC),
        ("nu", 0x3BD), ("xi", 0x3BE), ("pi", 0x3C0), ("rho", 0x3C1),
        ("sigma", 0x3C3), ("tau", 0x3C4), ("phi", 0x3C6), ("chi", 0x3C7),
        ("psi", 0x3C8), ("omega", 0x3C9), ("Gamma", 0x393), ("Delta", 0x394),
        ("Theta", 0x398), ("Lambda", 0x39B), ("Pi", 0x3A0), ("Sigma", 0x3A3),
        ("Phi", 0x3A6), ("Psi", 0x3A8), ("Omega", 0x3A9),
    )
}


def segments(text: str) -> list[tuple[str, bool]]:
    """The parts of a text in order.

    Args:
        text: A text, possibly with ``$...$`` parts.

    Returns:
        list of tuple: ``(piece, is math)``; one plain piece if the
            number of ``$`` is odd.
    """
    pieces = text.split("$")
    if len(pieces) % 2 == 0:
        return [(text, False)]
    return [(piece, k % 2 == 1) for k, piece in enumerate(pieces) if piece]


def as_plain(text: str | None) -> str | None:
    """`text` without math markers: Greek letters (``\\alpha``) as
    characters, other commands and braces dropped.

    Args:
        text: A text, possibly with ``$...$`` parts.

    Returns:
        str or None: E.g. ``"x_1 = 0.5"`` for ``"$x_1$ = 0.5"``.
    """
    if not text:
        return text
    return "".join(
        symbols(piece).replace("{", "").replace("}", "") if is_math else piece
        for piece, is_math in segments(text)
    )


def symbols(math: str) -> str:
    """A math part with Greek letters (``\\alpha``) as characters, spacing
    commands as spaces and other commands without their backslash.

    Args:
        math: A math part, without ``$``.

    Returns:
        str: The part with symbols replaced.
    """
    math = re.sub(r"\\([A-Za-z]+)", lambda m: GREEK.get(m.group(1), m.group(1)), math)
    return math.replace(r"\,", " ").replace(r"\ ", " ").replace("\\", "")
