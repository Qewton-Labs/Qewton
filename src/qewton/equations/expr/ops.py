"""Operators and their registry.

An `Operator` says how many expression arguments it takes, whether derivative axes
follow them, its derivative order, its linearity, and how shapes go through it. Each
kind of operator is a subclass (`Gradient`, `Sum`, ...) with these as class
attributes and methods; the operators themselves are instances in the registry.

`OPS` holds every operator: by name (`OPS["grad"]`, e.g. to load saved expressions),
and, for the built-in ones, as attributes (`OPS.grad`), which is how code refers to
them. Calling an operator builds the expression: `OPS.grad(u, x)`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from qewton.config.variables import Variable

if TYPE_CHECKING:
    from qewton.equations.expr.base import Expr

Shape = tuple[int, ...]


class ShapeError(ValueError):
    """Raised when the arguments of an operator have incompatible shapes."""


class AxisError(ValueError):
    """Raised when an expression or constraint uses coordinates inconsistently."""


class Linearity(str, Enum):
    """How an expression depends on a set of unknowns.

    - NONE: no unknown occurs.
    - LINEAR: a linear function of the unknowns, e.g. nu * grad(u).
    - AFFINE: linear plus something that does not depend on them, e.g. u + 1.
    - NONLINEAR: anything else, e.g. u * u or sin(u).
    """

    NONE = "none"
    LINEAR = "linear"
    AFFINE = "affine"
    NONLINEAR = "nonlinear"

    def __str__(self) -> str:
        return self.value

    def plus(self, other: Linearity) -> Linearity:
        """The dependence of the sum of two expressions."""
        if Linearity.NONLINEAR in (self, other):
            return Linearity.NONLINEAR
        if self is Linearity.NONE:
            return other
        if other is Linearity.NONE:
            return self
        return Linearity.LINEAR if self is other is Linearity.LINEAR else Linearity.AFFINE


class OpLinearity(str, Enum):
    """How an operator depends on its arguments.

    - LINEAR: linear in every argument jointly, like a sum or a derivative.
    - MULTILINEAR: linear in each argument alone, like a product.
    - NONLINEAR: neither, like sin.
    """

    LINEAR = "linear"
    MULTILINEAR = "multilinear"
    NONLINEAR = "nonlinear"

    def __str__(self) -> str:
        return self.value

    def of_result(self, arguments: list[Linearity]) -> Linearity:
        """The dependence of the result, given that of each expression argument."""
        involved = [part for part in arguments if part is not Linearity.NONE]
        if not involved:
            return Linearity.NONE
        if self is OpLinearity.NONLINEAR:
            return Linearity.NONLINEAR
        if self is OpLinearity.MULTILINEAR:
            return Linearity.NONLINEAR if len(involved) > 1 else involved[0]
        result = Linearity.NONE
        for part in arguments:
            result = result.plus(part)
        # A term without unknowns next to one with is a constant offset.
        if len(involved) < len(arguments) and result is Linearity.LINEAR:
            return Linearity.AFFINE
        return result


class Solver(Protocol):
    """What operators and nodes may ask of shape inference."""

    def known(self, expr: Any) -> Shape | None:
        """The shape of `expr` if it is already determined, else None."""

    def back(self, expr: Any, target: Shape) -> None:
        """Record that `expr` has shape `target`."""

    def record(self, symbol: Any, shape: Shape) -> None:
        """Record that the function `symbol`, whose shape is not declared, has
        `shape`; raises ShapeError if it already has another one."""


class Operator:
    """One operator. A kind of operator is a subclass that states:

    - `arity`: the number of expression arguments;
    - `takes_axes`: whether a coordinate product follows them, e.g. x in grad(u, x);
    - `order`: the derivative order along each coordinate of those axes;
    - `linearity`: how it depends on its arguments;
    - `result_shape`: the result shape from the argument shapes;
    - `infer_operands`: what a known result shape says about the arguments, for
      shape inference; by default nothing;
    - `require_operands`: shapes the arguments need whatever the result; by default
      none.

    An operator defined as an attribute of `OperatorRegistry` takes the attribute's
    name; one defined elsewhere is given its name and registered.
    """

    arity: int = 1
    takes_axes: bool = False
    order: int = 0
    linearity: OpLinearity = OpLinearity.LINEAR

    def __init__(self, name: str = "") -> None:
        self.name = name

    def __set_name__(self, owner: type, name: str) -> None:
        """Take the name of the class attribute this operator is defined as."""
        if self.name and self.name != name:
            raise ValueError(f"The operator {self.name!r} is defined as {name!r}.")
        self.name = name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    def __call__(self, *args: object) -> Expr:
        """The expression applying this operator, e.g. `OPS.grad(u, x)`. Python
        numbers and one-dimensional coordinates are accepted as expressions; the
        derivative axes follow the expression arguments as they are."""
        # Apply and the leaves build on this module, so they are imported on use.
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from qewton.equations.expr.apply import Apply

        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from qewton.equations.expr.leaves import as_expr

        operands = tuple(as_expr(arg) for arg in args[: self.arity])
        return Apply(self, operands + args[self.arity :])

    def result_shape(  # pylint: disable=unused-argument
        self, shapes: tuple[Shape, ...], axes: Variable | None
    ) -> Shape:
        """The result shape; raises ShapeError if the argument shapes do not fit.
        By default the shape of the first argument."""
        return shapes[0]

    def infer_operands(
        self,
        operands: tuple[Expr, ...],
        axes: Variable | None,
        target: Shape,
        solver: Solver,
    ) -> None:
        """Record what a result of shape `target` says about the operands."""

    def require_operands(self, operands: tuple[Expr, ...], solver: Solver) -> None:
        """Record the shapes the operands need, whatever the result."""


class BinaryOperator(Operator):
    """An operator of two expression arguments."""

    arity = 2

    @staticmethod
    def _with_other(operands: tuple[Expr, ...]):
        """(operand, the other operand), for both operands."""
        first, second = operands
        return (first, second), (second, first)


class Sum(BinaryOperator):
    """u + v or u - v: equal shapes, where a scalar fits any shape."""

    def result_shape(self, shapes, axes):
        first, second = shapes
        if second in (first, ()):
            return first
        if first == ():
            return second
        raise ShapeError(f"Cannot combine shapes {first} and {second} elementwise.")

    def infer_operands(self, operands, axes, target, solver):
        # With the other side scalar, or a scalar result, the unknown side is the
        # result. Otherwise it could be a broadcast scalar, so nothing is decided.
        for operand, other in self._with_other(operands):
            shape = solver.known(other)
            if shape is not None and solver.known(operand) is None:
                if shape == () or not target:
                    solver.back(operand, target)


class Product(BinaryOperator):
    """u * v: one factor is a scalar, and the result has the shape of the other."""

    linearity = OpLinearity.MULTILINEAR

    def result_shape(self, shapes, axes):
        first, second = shapes
        if first == ():
            return second
        if second == ():
            return first
        raise ShapeError(
            f"mul needs one scalar factor, got {first} and {second}. "
            "Use dot or inner for two vectors."
        )

    def infer_operands(self, operands, axes, target, solver):
        # The unknown factor is the result if the other is a scalar, else a scalar.
        for operand, other in self._with_other(operands):
            shape = solver.known(other)
            if shape is not None and solver.known(operand) is None:
                solver.back(operand, target if shape == () else ())


class Contraction(BinaryOperator):
    """dot(A, b): the last axis of the first argument against the first axis of the
    second. The result does not say how its axes split between the arguments, so
    nothing is inferred."""

    linearity = OpLinearity.MULTILINEAR

    def result_shape(self, shapes, axes):
        first, second = shapes
        if not first or not second:
            raise ShapeError(
                f"dot needs two non-scalar arguments, got {first} and {second}."
            )
        if first[-1] != second[0]:
            raise ShapeError(
                f"dot contracts the last axis of {first} with the first axis of {second}."
            )
        return first[:-1] + second[1:]


class InnerProduct(BinaryOperator):
    """inner(u, v): equal shapes, contracted to a scalar."""

    linearity = OpLinearity.MULTILINEAR

    def result_shape(self, shapes, axes):
        first, second = shapes
        if first != second:
            raise ShapeError(f"inner needs equal shapes, got {first} and {second}.")
        return ()

    def infer_operands(self, operands, axes, target, solver):
        for operand, other in self._with_other(operands):
            shape = solver.known(other)
            if shape is not None and solver.known(operand) is None:
                solver.back(operand, shape)


class ScalarFunction(Operator):
    """sin(u), exp(u), u**v: scalar arguments and a scalar result."""

    linearity = OpLinearity.NONLINEAR

    def __init__(self, arity: int = 1, name: str = "") -> None:
        super().__init__(name)
        self.arity = arity

    def result_shape(self, shapes, axes):
        if any(shape != () for shape in shapes):
            raise ShapeError(f"Expected scalar arguments, got shapes {shapes}.")
        return ()

    def require_operands(self, operands, solver):
        for operand in operands:
            solver.back(operand, ())


class ShapePreserving(Operator):
    """An operator whose result has the shape of its operand, e.g. -v."""

    def infer_operands(self, operands, axes, target, solver):
        solver.back(operands[0], target)


class PartialDerivative(ShapePreserving):
    """d(u, t): the derivative along one coordinate."""

    takes_axes = True
    order = 1

    def result_shape(self, shapes, axes):
        if axes is not None and axes.dim != 1:
            raise ShapeError("d needs a single coordinate; use grad for several.")
        return shapes[0]


class Laplacian(ShapePreserving):
    """lap(u, x): the sum of the second derivatives, of the shape of u."""

    takes_axes = True
    order = 2


class Gradient(Operator):
    """grad(u, x): a new last axis of the length of the coordinates."""

    takes_axes = True
    order = 1

    def result_shape(self, shapes, axes):
        return shapes[0] + (axes.dim,)  # type: ignore[union-attr]

    def infer_operands(self, operands, axes, target, solver):
        dim = axes.dim  # type: ignore[union-attr]
        if not target or target[-1] != dim:
            raise ShapeError(f"a gradient along {dim} ends in {dim}, got {target}.")
        solver.back(operands[0], target[:-1])


class Divergence(Operator):
    """div(v, x): the last axis, of the length of the coordinates, is summed away."""

    takes_axes = True
    order = 1

    def result_shape(self, shapes, axes):
        shape, dim = shapes[0], axes.dim  # type: ignore[union-attr]
        if not shape or shape[-1] != dim:
            raise ShapeError(f"div needs a last axis of length {dim}, got shape {shape}.")
        return shape[:-1]

    def infer_operands(self, operands, axes, target, solver):
        solver.back(operands[0], target + (axes.dim,))  # type: ignore[union-attr]


class Curl(Operator):
    """curl(v, x): a 3-vector along three coordinates."""

    takes_axes = True
    order = 1

    def result_shape(self, shapes, axes):
        if shapes[0] != (3,) or axes is None or axes.dim != 3:
            raise ShapeError("curl needs a 3-vector and three coordinates.")
        return (3,)


class OperatorRegistry(Mapping[str, Operator]):
    """Every operator by name. The built-in ones are also attributes: `OPS.grad`.

    Code refers to an operator by its attribute (`expr.op is OPS.lap`), never by its
    name, so pylint and editors flag a misspelled one. A built-in operator is one more
    attribute below, and its name is the attribute's name. An operator defined
    elsewhere is registered once:

        TANH = OPS.register(ScalarFunction(name="tanh"))

    after which it is also `OPS["tanh"]` and `OPS.tanh`.
    """

    add = Sum()
    sub = Sum()
    neg = ShapePreserving()
    mul = Product()
    dot = Contraction()
    inner = InnerProduct()
    pow = ScalarFunction(2)
    exp = ScalarFunction()
    sin = ScalarFunction()
    cos = ScalarFunction()
    d = PartialDerivative()
    grad = Gradient()
    div = Divergence()
    curl = Curl()
    lap = Laplacian()

    def __init__(self) -> None:
        self._operators: dict[str, Operator] = {
            op.name: op for op in vars(type(self)).values() if isinstance(op, Operator)
        }

    def register(self, op: Operator) -> Operator:
        """Add an operator defined outside this class, and return it."""
        if not op.name:
            raise ValueError(
                "An operator needs a name, e.g. ScalarFunction(name='tanh')."
            )
        if op.name in self._operators:
            raise ValueError(f"Operator {op.name!r} is already registered.")
        if hasattr(self, op.name):
            raise ValueError(f"{op.name!r} cannot name an operator: OPS uses it.")
        self._operators[op.name] = op
        setattr(self, op.name, op)
        return op

    def __getitem__(self, name: str) -> Operator:
        return self._operators[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._operators)

    def __len__(self) -> int:
        return len(self._operators)


OPS = OperatorRegistry()
