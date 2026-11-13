"""Symbolic PDE problems: geometry, expressions, constraints, analysis and weak forms.

    import qewton.equations as eq

    x = eq.Variable("x", dim=2)
    Omega = eq.Domain("Omega", ambient=x)
    u = eq.FunctionSymbol(eq.Variable("u", dim=1), x)
    wall = eq.Boundary(Omega)
    problem = eq.Problem(
        (
            eq.Eq(-eq.lap(u, x), 1).on(Omega),
            eq.Eq(eq.Restrict(u, wall), 0).on(wall),
        )
    )

    ctx = eq.AnalysisContext(problem.task(unknown=[u]))
    print(ctx.report())

The package is layered; each layer builds only on the ones before it, and its
`__init__.py` lists its public names, which are all available from here:

    geometry   the sets constraints hold on
    expr       expressions and the operator registry
    model      constraints, Problem and Task
    analysis   AnalysisContext and everything it finds out
    compile    the weak form
    printing   readable text and the coupling diagram
"""

# The layers list their public names in their own __all__.
# pylint: disable=wildcard-import,unused-wildcard-import,redefined-builtin
from qewton.config.variables import Variable
from qewton.equations import analysis, compile, expr, geometry, model, printing
from qewton.equations.analysis import *
from qewton.equations.compile import *
from qewton.equations.diagnostics import Diagnostic, Severity
from qewton.equations.expr import *
from qewton.equations.geometry import *
from qewton.equations.model import *
from qewton.equations.printing import *

__all__ = ["Variable", "Diagnostic", "Severity"]
__all__ += geometry.__all__
__all__ += expr.__all__
__all__ += model.__all__
__all__ += analysis.__all__
__all__ += compile.__all__
__all__ += printing.__all__
