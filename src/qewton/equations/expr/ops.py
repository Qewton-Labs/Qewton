"""Operator registry: operators as data.

Each OpSpec says how many expression arguments an operator takes, whether it takes
derivative axes after them, its derivative order, its linearity, and how the shape
of its result relates to the shapes of its arguments (`relate`).

Shapes are related as data configurations (`qewton.config`), the description the
computation graphs use for the data between nodes. A relation is one function in both
directions: it reads what is known of the arguments, constrains them, and returns the
result, all through objects that are merged once two places are known to be the same.
So a known result shape flows back into the arguments without a second rule, e.g. a
scalar `div(q, x)` makes `q` a vector of length dim(x).

`OPS` holds every operator: by name (`OPS["grad"]`, e.g. to load saved expressions),
and, for the built-in ones, as attributes (`OPS.grad`), which is how code refers to
them; see `OperatorRegistry`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum

from qewton.config.axes import Axes, AxesDim, EllipsisAxes
from qewton.config.data_configurations.base import DataConfiguration
from qewton.config.errors import DataConfigMismatchError, StructureUnknown
from qewton.config.variables import Variable

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


###
# Shapes as data configurations: one axes per dim, an ellipsis for a rest not
# known yet. A known shape is a tuple everywhere else.
###
def fixed(shape: Shape) -> DataConfiguration:
    """A known shape as a configuration; () is a scalar."""
    return DataConfiguration(*(Axes(size) for size in shape))


def unknown() -> DataConfiguration:
    """A shape of which nothing is known yet."""
    return DataConfiguration(EllipsisAxes())


def known_rank(config: DataConfiguration) -> int | None:
    """The number of dims, or None while it is open."""
    shape = config.resolve().shape
    return None if Ellipsis in shape else len(shape)


def known_shape(config: DataConfiguration) -> Shape | None:
    """The shape, or None while the number of dims or a size is open."""
    shape = config.resolve().shape
    return None if Ellipsis in shape or None in shape else shape  # type: ignore


def describe(config: DataConfiguration) -> str:
    """The shape for messages: (2, 3), (…, 2) for an open rest, ? for an open size."""
    parts = [
        "…" if s is Ellipsis else "?" if s is None else str(s)
        for s in config.resolve().shape
    ]
    return f"({parts[0]},)" if len(parts) == 1 else f"({', '.join(parts)})"


def unify(first, second, message: str) -> None:
    """Merge two configurations, or two dims; raise ShapeError(message) if they
    contradict."""
    try:
        first.merge(second)
    except DataConfigMismatchError as error:
        raise ShapeError(message) from error


# A relation: the configurations of the expression arguments and the derivative axes
# (or None) in, the configuration of the result out. It may constrain the arguments;
# it raises ShapeError on a contradiction, and StructureUnknown while it cannot decide
# yet, e.g. which side of a sum is the scalar.
Relation = Callable[[list[DataConfiguration], "Variable | None"], DataConfiguration]


@dataclass(frozen=True)
class OpSpec:
    """The semantics of one operator.

    Attributes:
        arity: The number of expression arguments.
        relate: How the result shape relates to the argument shapes; see `Relation`.
        takes_axes: Whether a coordinate product follows the expression arguments.
        order: The derivative order along each leaf of the axes argument.
        linearity: How the operator depends on its arguments.
        name: The registry key, e.g. "grad". An operator defined as an attribute of
            `OperatorRegistry` takes the attribute's name, so it is not given.
    """

    arity: int
    relate: Relation
    takes_axes: bool = False
    order: int = 0
    linearity: OpLinearity = OpLinearity.LINEAR
    name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        """Take the name of the class attribute this operator is defined as."""
        if self.name and self.name != name:
            raise ValueError(f"The operator {self.name!r} is defined as {name!r}.")
        object.__setattr__(self, "name", name)

    def result_shape(
        self, shapes: tuple[Shape, ...], axes: Variable | None = None
    ) -> Shape:
        """The result shape for known argument shapes, e.g. (2,) for grad(u, x) with a
        scalar u and a two-dimensional x."""
        result = known_shape(self.relate([fixed(shape) for shape in shapes], axes))
        assert result is not None, "known arguments give a known result"
        return result


###
# The relations of the built-in operators.
###
def _same(configs: list[DataConfiguration], _axes: Variable | None) -> DataConfiguration:
    return configs[0]


def _sum(configs: list[DataConfiguration], _axes: Variable | None) -> DataConfiguration:
    """Equal shapes, where a scalar fits any shape."""
    first, second = configs
    if known_rank(first) == 0:
        return second
    if known_rank(second) == 0:
        return first
    if known_rank(first) is None or known_rank(second) is None:
        raise StructureUnknown("Which side of the sum is a scalar is not known yet.")
    message = (
        f"Cannot combine shapes {describe(first)} and {describe(second)} elementwise."
    )
    unify(first, second, message)
    return first


def _product(
    configs: list[DataConfiguration], _axes: Variable | None
) -> DataConfiguration:
    """One factor is a scalar; the result has the shape of the other."""
    first, second = configs
    ranks = known_rank(first), known_rank(second)
    if ranks[0] == 0:
        return second
    if ranks[1] == 0:
        return first
    if None not in ranks:
        raise ShapeError(
            f"mul needs one scalar factor, got {describe(first)} and {describe(second)}. "
            "Use dot or inner for two vectors."
        )
    if ranks[0] is None and ranks[1] is None:
        raise StructureUnknown("Which factor is a scalar is not known yet.")
    factor, other = (second, first) if ranks[0] is not None else (first, second)
    unify(factor, fixed(()), f"mul needs one scalar factor, got {describe(factor)}.")
    return other


def _contract(
    configs: list[DataConfiguration], _axes: Variable | None
) -> DataConfiguration:
    """The last axis of the first argument against the first axis of the second."""
    first, second = configs
    if 0 in (known_rank(first), known_rank(second)):
        raise ShapeError(
            f"dot needs two non-scalar arguments, got {describe(first)} and "
            f"{describe(second)}."
        )
    message = (
        f"dot contracts the last axis of {describe(first)} with the first axis of "
        f"{describe(second)}."
    )
    unify(first.dim(-1), second.dim(0), message)
    return DataConfiguration(*first.drop(-1).axes, *second.drop(0).axes)


def _inner(configs: list[DataConfiguration], _axes: Variable | None) -> DataConfiguration:
    first, second = configs
    message = f"inner needs equal shapes, got {describe(first)} and {describe(second)}."
    unify(first, second, message)
    return fixed(())


def _scalars(
    configs: list[DataConfiguration], _axes: Variable | None
) -> DataConfiguration:
    for config in configs:
        unify(config, fixed(()), f"Expected scalar arguments, got {describe(config)}.")
    return fixed(())


def scalar_op(arity: int, name: str = "") -> OpSpec:
    """A nonlinear operator whose arguments and result are all scalars."""
    return OpSpec(arity, _scalars, linearity=OpLinearity.NONLINEAR, name=name)


def _partial(
    configs: list[DataConfiguration], axes: Variable | None
) -> DataConfiguration:
    if axes is None or axes.dim != 1:
        raise ShapeError("d needs a single coordinate; use grad for several.")
    return configs[0]


def _grad(configs: list[DataConfiguration], axes: Variable | None) -> DataConfiguration:
    """A new last axis of the length of the coordinates."""
    return DataConfiguration(*configs[0].axes, Axes(axes.dim))  # type: ignore[union-attr]


def _div(configs: list[DataConfiguration], axes: Variable | None) -> DataConfiguration:
    """The last axis, of the length of the coordinates, is summed away."""
    (field,) = configs
    dim = axes.dim  # type: ignore[union-attr]
    message = f"div needs a last axis of length {dim}, got shape {describe(field)}."
    if known_rank(field) == 0:
        raise ShapeError(message)
    unify(field.dim(-1), AxesDim(dim), message)
    return field.drop(-1)


def _curl(configs: list[DataConfiguration], axes: Variable | None) -> DataConfiguration:
    (field,) = configs
    message = "curl needs a 3-vector and three coordinates."
    if axes is None or axes.dim != 3:
        raise ShapeError(message)
    unify(field, fixed((3,)), message)
    return field


class OperatorRegistry(Mapping[str, OpSpec]):
    """Every operator by name. The built-in ones are also attributes: `OPS.grad`.

    Code refers to an operator by its attribute (`expr.op is OPS.lap`), never by its
    name, so pylint and editors flag a misspelled one. A built-in operator is one more
    attribute below, and its name is the attribute's name. An operator defined
    elsewhere is registered once:

        TANH = OPS.register(scalar_op(1, name="tanh"))

    after which it is also `OPS["tanh"]` and `OPS.tanh`.
    """

    add = OpSpec(2, _sum)
    sub = OpSpec(2, _sum)
    neg = OpSpec(1, _same)
    mul = OpSpec(2, _product, linearity=OpLinearity.MULTILINEAR)
    dot = OpSpec(2, _contract, linearity=OpLinearity.MULTILINEAR)
    inner = OpSpec(2, _inner, linearity=OpLinearity.MULTILINEAR)
    pow = scalar_op(2)
    exp = scalar_op(1)
    sin = scalar_op(1)
    cos = scalar_op(1)
    d = OpSpec(1, _partial, takes_axes=True, order=1)
    grad = OpSpec(1, _grad, takes_axes=True, order=1)
    div = OpSpec(1, _div, takes_axes=True, order=1)
    curl = OpSpec(1, _curl, takes_axes=True, order=1)
    lap = OpSpec(1, _same, takes_axes=True, order=2)

    def __init__(self) -> None:
        self._specs: dict[str, OpSpec] = {
            spec.name: spec
            for spec in vars(type(self)).values()
            if isinstance(spec, OpSpec)
        }

    def register(self, spec: OpSpec) -> OpSpec:
        """Add an operator defined outside this class, and return it."""
        if not spec.name:
            raise ValueError("An operator needs a name, e.g. OpSpec(..., name='tanh').")
        if spec.name in self._specs:
            raise ValueError(f"Operator {spec.name!r} is already registered.")
        if hasattr(self, spec.name):
            raise ValueError(f"{spec.name!r} cannot name an operator: OPS uses it.")
        self._specs[spec.name] = spec
        setattr(self, spec.name, spec)
        return spec

    def __getitem__(self, name: str) -> OpSpec:
        return self._specs[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)


OPS = OperatorRegistry()
