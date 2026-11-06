"""Texts as Plotly displays them.

The core joins plain text and math freely, e.g. ``"$t$ = 0.5"`` (see
:mod:`qewton.visualization.renderers.text`). Plotly renders math only
through MathJax, only for a text that is math as a whole, and never in
slider labels or dropdown buttons. :func:`as_math`
turns a text into one math text for MathJax; :func:`as_markup` turns it
into Plotly's own text markup (italics, sub- and superscripts), which
displays everywhere in a Plotly figure. Plain text, e.g. for the options
of a Dash widget, comes from
:func:`~qewton.visualization.renderers.text.as_plain`.
"""

import html
import re

from qewton.visualization.renderers.text import as_plain, segments, symbols  # noqa: F401

# A sub- or superscript: a braced group or a single character.
_SCRIPT = re.compile(r"([_^])(\{[^{}]*\}|.)")


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
    pieces = segments(text)
    if not any(is_math for _, is_math in pieces):
        return text
    parts = [
        piece if is_math else r"\text{" + piece.replace("}", r"\}") + "}"
        for piece, is_math in pieces
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
        for piece, is_math in segments(text)
    )


def _math_markup(math: str) -> str:
    math = symbols(math)
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
