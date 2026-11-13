"""Compilers from the strong form to other representations; so far the weak form.

    weak = weak_form(task)

`compiler.py` builds the `WeakProblem` of `forms.py`, using integration by parts
(`by_parts.py`), the pairing of equations and test functions (`pairing.py`) and the
treatment of boundary terms (`boundary.py`).
"""

from qewton.equations.compile.compiler import weak_form
from qewton.equations.compile.forms import FormTerm, Measure, Residual, WeakProblem

__all__ = ["FormTerm", "Measure", "Residual", "WeakProblem", "weak_form"]
