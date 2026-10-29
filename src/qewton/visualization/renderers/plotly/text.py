"""Texts as Plotly displays them.

The core writes names of Variables in math mode (``$x_1$``), everything
else as plain text, and joins them freely, e.g. ``"$t$ = 0.5"``. Plotly
renders math only through MathJax, only for a text that is math as a
whole, and never in slider labels or dropdown buttons. :func:`as_math`
turns a text into one math text for MathJax; :func:`as_markup` turns it
into Plotly's own text markup (italics, sub- and superscripts), which
displays everywhere in a Plotly figure; :func:`as_plain` into plain text,
e.g. for the options of a Dash widget.
"""

import html
import re

_GREEK = {
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

# A sub- or superscript: a braced group or a single character.
_SCRIPT = re.compile(r"([_^])(\{[^{}]*\}|.)")


def _segments(text: str) -> list[tuple[str, bool]]:
    # (piece, is math) in order; a text with an odd number of "$" is
    # plain as a whole.
    pieces = text.split("$")
    if len(pieces) % 2 == 0:
        return [(text, False)]
    return [(piece, k % 2 == 1) for k, piece in enumerate(pieces) if piece]


def as_math(text: str | None) -> str | None:
    """`text` as one math text, with its plain parts as ``\\text{...}``;
    unchanged without math.

    Args:
        text: A text, possibly with ``$...$`` parts.

    Returns:
        str or None: E.g. ``"$t\\text{ = 0.5}$"`` for ``"$t$ = 0.5"``.
    """
    if not text:
        return text
    segments = _segments(text)
    if not any(is_math for _, is_math in segments):
        return text
    parts = [
        piece if is_math else r"\text{" + piece.replace("}", r"\}") + "}"
        for piece, is_math in segments
    ]
    return "$" + "".join(parts) + "$"


def as_markup(text: str | None) -> str | None:
    """`text` in Plotly's text markup: math parts italic, with sub- and
    superscripts and Greek letters (``\\alpha``); plain parts unchanged.

    Args:
        text: A text, possibly with ``$...$`` parts.

    Returns:
        str or None: E.g. ``"<i>x</i><sub>1</sub> = 0.5"`` for
            ``"$x_1$ = 0.5"``.
    """
    if not text:
        return text
    return "".join(
        _math_markup(piece) if is_math else html.escape(piece, quote=False)
        for piece, is_math in _segments(text)
    )


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
        _symbols(piece).replace("{", "").replace("}", "") if is_math else piece
        for piece, is_math in _segments(text)
    )


def _symbols(math: str) -> str:
    # Greek letters as characters, spacing commands as spaces, other
    # commands without their backslash.
    math = re.sub(r"\\([A-Za-z]+)", lambda m: _GREEK.get(m.group(1), m.group(1)), math)
    return math.replace(r"\,", " ").replace(r"\ ", " ").replace("\\", "")


def _math_markup(math: str) -> str:
    math = _symbols(math)
    parts, end = [], 0
    for match in _SCRIPT.finditer(math):
        parts.append(_italic(math[end : match.start()]))
        tag = "sub" if match.group(1) == "_" else "sup"
        script = match.group(2).strip("{}")
        parts.append(f"<{tag}>{html.escape(script, quote=False)}</{tag}>")
        end = match.end()
    parts.append(_italic(math[end:]))
    return "".join(parts)


def _italic(piece: str) -> str:
    # Letters italic, as in math; digits, signs and spaces upright.
    piece = html.escape(piece.replace("{", "").replace("}", ""), quote=False)
    return re.sub(r"([^\W\d_]+)", r"<i>\1</i>", piece)
