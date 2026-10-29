"""Names of references into data: Variables and Axes of a
DataConfiguration, or names of table columns, as shown in titles."""

from qewton.config.axes import Axes, GeometryAxes
from qewton.config.variables import Variable


def title_of(reference: Variable | Axes | str) -> str | None:
    """The title naming a reference.

    Args:
        reference: A Variable, an Axes of a DataConfiguration, or the name
            of a table column.

    Returns:
        str or None: The Variable's math name (for a GeometryAxes, the one
            of its geometry's variable); a column name as it is; None for
            an Axes without a variable, e.g. a BatchAxes.
    """
    if isinstance(reference, str):
        return reference
    if isinstance(reference, Variable):
        return None if reference.is_empty else reference.math_name
    if isinstance(reference, GeometryAxes):
        variable = reference.geometry.variable
        return None if variable is None or variable.is_empty else variable.math_name
    return None


def component_titles(reference: Variable | Axes, n: int) -> list[str | None]:
    """Titles of the `n` components of a reference, e.g. ``$x$`` and
    ``$y$`` for ``Variable("x", 1) * Variable("y", 1)``, ``$x_1$`` and
    ``$x_2$`` for ``Variable("x", 2)``.

    Args:
        reference: A Variable, or an Axes of a DataConfiguration.
        n: Number of components.

    Returns:
        list of str or None: One title per component; None where the
            reference has no name.
    """
    variable = (
        reference.geometry.variable if isinstance(reference, GeometryAxes) else reference
    )
    if n == 1:
        return [title_of(reference)]
    if not isinstance(variable, Variable) or variable.is_empty:
        return [None] * n
    leaves = variable.leaves
    if len(leaves) == n:
        return [leaf.math_name for leaf in leaves]
    return [f"${variable.name}_{i + 1}$" for i in range(n)]
