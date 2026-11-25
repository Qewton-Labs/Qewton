from __future__ import annotations

from copy import deepcopy

from typing import Any

import math
from types import EllipsisType

from qewton.config.variables import Variable
from qewton.config.errors import DataConfigMismatchError
from qewton.config.saving.saving import Serializable


def _contradiction(first: AxesDim, second: AxesDim) -> str:
    """The message for two dims of different sizes that should be one."""

    def described(dim: AxesDim) -> str:
        sources = [f"from {dim.origin}"] if dim.origin is not None else []
        sources += [f"HyperParameter {p.name!r}" for p in dim.hyperparameters]
        return f"{dim.size} ({', '.join(sources)})" if sources else str(dim.size)

    return f"Sizes {described(first)} and {described(second)} do not match."


class Axes(Serializable):
    """
    Represents a collection of axes (dimensions) for data, including their
    sizes and types.

    An `Axes` of this base class is an axes of unknown kind; merging it with
    a `BatchAxes`, `GeometryAxes` or `FeatureAxes` gives that kind. Merging
    (:meth:`merge`) makes two axes one, as for dims (see
    :class:`AxesDim`).

    Attributes:
        merged_into (Axes | None): The axes this one was merged into; None
            for a representative.
    """

    merged_into: Axes | None = None
    #: Whether this stands for a sequence of axes (an ellipsis).
    is_ellipsis = False

    @property
    def representative(self) -> Axes:
        """The axes that stands for this one and all merged into it."""
        if self.merged_into is None:
            return self
        representative = self.merged_into.representative
        self.merged_into = representative
        return representative

    def merge(self, other: Axes) -> Axes:
        """Make this axes and `other` one.

        The more specific of the two stays: a known kind, then a Variable or
        a geometry. Its dims are merged with the other's.

        Args:
            other: The axes to merge with.

        Returns:
            Axes: The representative of both.

        Raises:
            DataConfigMismatchError: If the kinds, contents or dims of the
                axes contradict.
        """
        from qewton.config.sequences import merge_sequences

        first, second = self.representative, other.representative
        if first is second:
            return first
        if not first._kind_compatible(second) or not first._content_compatible(second):
            raise DataConfigMismatchError(f"{first} and {second} can not be one axes.")
        keep, gone = (first, second) if first._outranks(second) else (second, first)
        keep._take_content(gone)
        gone.merged_into = keep
        merge_sequences(keep._shape, gone._shape)
        return keep

    def compatible(self, other: Axes) -> bool:
        """Whether this axes and `other` could be merged."""
        from qewton.config.sequences import sequences_compatible

        first, second = self.representative, other.representative
        return (
            first._kind_compatible(second)
            and first._content_compatible(second)
            and sequences_compatible(first._shape, second._shape)
        )

    def _kind_compatible(self, other: Axes) -> bool:
        return type(self) is Axes or type(other) is Axes or type(self) is type(other)

    def _content_compatible(self, other: Axes) -> bool:
        """Whether the contents (e.g. Variables) of both axes agree; axes of
        different kinds (one of them unknown) have nothing to compare."""
        return type(self) is not type(other) or self._contents_agree(other)

    def _contents_agree(self, other: Axes) -> bool:
        """Whether the contents of this axes and `other`, of the same kind,
        agree."""
        return True

    def _outranks(self, other: Axes) -> bool:
        """Whether this axes stays when merged with `other`."""
        if type(other) is Axes and type(self) is not Axes:
            return True
        return type(self) is type(other) and self._specificity() > other._specificity()

    def _specificity(self) -> int:
        """How much this axes knows beyond its kind, e.g. a Variable."""
        return 0

    def _take_content(self, other: Axes) -> None:
        """Take over what `other` knows beyond this axes, before `other` is
        merged into this one."""

    def with_dims(self, dims) -> Axes:
        """New axes of the same kind with `dims`, e.g. after an operation
        removed or changed one of them."""
        return type(self)(*dims)

    def __init__(self, *shape: int | AxesDim | None | EllipsisType):
        new_shape = []
        for s in shape:
            if isinstance(s, int) or s is None:
                new_shape.append(AxesDim(size=s))
            elif isinstance(s, EllipsisType):
                new_shape.append(EllipsisDim())
            else:
                new_shape.append(s)
        self._shape = tuple(new_shape)

    @property
    def shape(self):
        """
        Returns the tuple of `AxesDim` objects representing the shape.
        """
        return self._shape

    @property
    def is_empty(self):
        """
        Checks if the axes collection is empty.

        Returns:
            bool: True if the shape has no dimensions, False otherwise.
        """
        return len(self._shape) == 0

    def add_dim(self, new_dim, index):
        """Adds a new dimension to the axes at a specific index.

        Args:
            new_dim (AxesDim): The dimension to add.
            index (int): The index at which to add the dimension.
        """
        self._shape = self._shape[:index] + (new_dim,) + self._shape[index:]

    def matches(self, other: Axes) -> bool:
        """
        Checks if these axes exactly match another `Axes` object in terms of
        number of dimensions, type, and size.

        Args:
            other (Axes): The other `Axes` object to compare with.

        Returns:
            bool: True if the axes match, False otherwise.
        """
        if len(self.shape) != len(other.shape):
            return False
        # Iterate through dimensions and check for type and size equality.
        # This ensures a strict match.
        for s1, s2 in zip(self.shape, other.shape):
            if not isinstance(s1, s2.__class__):
                return False
            if not s1.size == s2.size:
                return False
        return True

    def __str__(self) -> str:
        """
        Returns a string representation of the Axes object.

        Returns:
            str: A string representation of the Axes, showing the class name and
                the shape.
        """
        return f"{self.__class__.__name__}([{', '.join(str(s) for s in self.shape)}])"

class BatchAxes(Axes):
    """
    Represents batch-specific axes, inheriting from `Axes`.

    This class can be used to explicitly denote dimensions that correspond to
    batch size.
    """


class GeometryAxes(Axes):
    """
    Represents geometry-specific axes, typically used for spatial dimensions.

    It encapsulates a `Geometry` object to manage the underlying geometric shape,
    which can also be used for plotting etc.
    """

    def __init__(
        self,
        geometry=None,
        shape: tuple[int | AxesDim, ...] | None = None,
    ):
        """Represents geometry-specific axes, typically used for
        spatial dimensions.

        Args:
            geometry (Geometry | None, optional): The geometry encoded in
                this axes object. Defaults to None.
            shape (tuple[int  |  AxesDim, ...] | None, optional): The expected shape
                the axes represents. Will create a dummy geometry, if no
                geometry is provided. Defaults to None.

        Raises:
            ValueError: Only one geometry or shape can be provided, not both.
            ValueError: Either geometry or shape must be provided.
        """
        from qewton.geometries.base import Geometry

        self._geometry: Geometry
        if geometry is not None:
            self._geometry = geometry
        elif shape is not None and geometry is None:
            self._geometry = Geometry(
                shape=tuple(i.size if isinstance(i, AxesDim) else i for i in shape)
            )
        else:
            raise ValueError("Either geometry or shape must be provided.")
        default_shape = (...,)
        if shape is not None:
            default_shape = shape
        elif not isinstance(self._geometry.shape, EllipsisType):
            default_shape = self._geometry.shape
        super().__init__(*default_shape)

    @property
    def geometry(self):
        """
        Returns the encapsulated `Geometry` object.

        Returns:
            Geometry: The geometry associated with these axes.
        """
        return self._geometry

    @property
    def variables(self):
        return self.geometry.variable

    def _is_placeholder(self) -> bool:
        """Whether the geometry only stands in for a shape (it has no
        Variable)."""
        variable = self._geometry.variable
        return variable is None or variable.is_empty

    def _specificity(self) -> int:
        return 0 if self._is_placeholder() else 1

    def _contents_agree(self, other: GeometryAxes) -> bool:
        mine, theirs = self._geometry, other._geometry
        return (
            mine is theirs
            or self._is_placeholder()
            or other._is_placeholder()
            or mine.is_discretization_of(theirs)
            or theirs.is_discretization_of(mine)
            or mine == theirs
        )

    def with_dims(self, dims) -> GeometryAxes:
        return GeometryAxes(shape=tuple(dims))

    def _take_content(self, other: Axes) -> None:
        if not isinstance(other, GeometryAxes) or other._is_placeholder():
            return
        if self._is_placeholder() or other._geometry.is_discretization_of(self._geometry):
            self._geometry = other._geometry

    def split(self, variables: list[Variable]) -> list[GeometryAxes]:
        """Split the axes of a product geometry into axes of its factors.

        Every factor whose variable is in `variables` gets axes of its own;
        each run of adjacent other factors stays together as their product.
        The tensor axes keep their order (the first factor varies slowest),
        so data described by these axes is reshaped, not reordered.

        Args:
            variables: Variables of factors of the geometry.

        Returns:
            list of GeometryAxes: In the order of the factors; together they
                have as many elements as these axes.

        Raises:
            ValueError: If the geometry is no product, a variable is no
                factor of it, or the factors do not span these axes.
        """
        factors = self.geometry.factors
        if factors is None:
            raise ValueError(f"{self.geometry} is not a product of geometries.")
        names = [f.variable for f in factors]
        unknown = [v for v in variables if v not in names]
        if unknown:
            raise ValueError(
                f"{', '.join(map(str, unknown))} is no factor of {self.geometry}; "
                f"its factors are {', '.join(map(str, names))}."
            )
        runs: list[list] = []
        for factor in factors:
            alone = factor.variable in variables
            if alone or not runs or runs[-1][0].variable in variables:
                runs.append([factor])
            else:
                runs[-1].append(factor)
        split = [
            GeometryAxes(run[0] if len(run) == 1 else self.geometry.subproduct(run))
            for run in runs
        ]
        size = math.prod(d.size for a in split for d in a.shape)
        own = math.prod(d.size for d in self.shape)
        if size != own:
            raise ValueError(
                f"The factors of {self.geometry.variable} have {size} points in "
                f"all, its axes {own}."
            )
        return split


class FeatureAxes(Axes):
    """
    Represents feature-specific axes, often associated with a `Variable` object
    that defines the features.
    """

    def __init__(
        self,
        variable: Variable | None = None,
        shape: tuple[int | AxesDim, ...] | None = None,
    ):
        if variable is not None and shape is not None:
            raise ValueError("Only one of variable or shape can be provided.")
        self._variable = None
        if variable is not None:
            self._variable = variable
            super().__init__(*self._variable.shape)
        elif shape is not None:
            super().__init__(*shape)
        else:
            super().__init__(None)

    @property
    def variables(self):
        """
        Returns the encapsulated `Variable` object, or a new empty `Variable` if none was
        set.
        """
        if self._variable is not None:
            return self._variable
        return Variable()

    def _specificity(self) -> int:
        return 0 if self._variable is None else 1

    def _contents_agree(self, other: FeatureAxes) -> bool:
        """The variables agree where they have the same components in the
        same order, e.g. ``x`` and its unnamed components ``(x_1, x_2)``."""
        mine, theirs = self._variable, other._variable
        if mine is None or theirs is None or mine is theirs:
            return True
        return [(leaf.name, leaf.dim) for leaf in mine.leaves] == [
            (leaf.name, leaf.dim) for leaf in theirs.leaves
        ]

    def with_dims(self, dims) -> FeatureAxes:
        return FeatureAxes(shape=tuple(dims))

    def _take_content(self, other: Axes) -> None:
        if isinstance(other, FeatureAxes) and self._variable is None:
            self._variable = other._variable

    def get_variable_slice(self, variable):
        """
        Retrieves a slice corresponding to a specific variable from the encapsulated
        `Variable` object.

        Args:
            variable: The variable for which to get the slice.

        Returns:
            Any: The slice corresponding to the variable.
        """
        return self.variables.get_slice(variable)

class EllipsisAxes(Axes):
    """
    Represents an ellipsis in axes, indicating a flexible number of dimensions.

    An ellipsis stands for a sequence of axes, possibly empty, until it is
    bound to one (see :mod:`qewton.config.sequences`).

    Attributes:
        bound_to (tuple[Axes, ...] | None): The sequence this ellipsis stands
            for; None while unknown.
        waiting (tuple): Equations of sequences to merge again once this
            ellipsis is bound.
    """

    is_ellipsis = True
    bound_to: tuple | None = None
    waiting: tuple = ()
    #: Operation dims over a sequence holding this ellipsis.
    _dependents: tuple = ()

    def compatible(self, other: Axes) -> bool:
        return True

    def __init__(self):
        super().__init__(EllipsisDim())

    def __str__(self) -> str:
        return "..."


class AxesDim(Serializable):
    """
    Represents a single dimension within a set of axes, with an optional size and
    broadcastability.

    A dim of unknown size is an unknown of matching. Merging two dims
    (:meth:`merge`) makes them one: the one merged away refers to the other,
    its `representative`, which stands for both from then on.

    A dim may be bound to a HyperParameter, `AxesDim(node.in_neurons)`: its
    size is the current value of the HyperParameter, also after it changed
    (e.g. a batch size), for every dim merged with it. An open one (state
    INFERRED, value None) is inferred once the dim's size is known; every
    copy of the dim (one per place of its node) refers to the same
    HyperParameter.

    Attributes:
        merged_into (AxesDim | None): The dim this one was merged into; None
            for a representative.
        origin (Any): What brought in the size, e.g. the port of a graph
            whose configuration declared it; used in error messages.
    """

    merged_into: AxesDim | None = None
    origin: Any = None
    #: Whether this stands for a sequence of dims (an ellipsis).
    is_ellipsis = False
    #: Operation dims with this dim as an operand.
    _dependents: tuple = ()
    #: Operation dims merged into this one while both sizes were unknown.
    _equal_operations: tuple = ()
    #: HyperParameters bound to this dim and to the dims merged into it.
    _parameters: tuple = ()

    def __new__(cls, size=None, broadcastable=True) -> AxesDim:
        """
        Creates a new `AxesDim` instance. If `size` is an `EllipsisType`, it returns an
        `EllipsisDim`.

        Args:
            size (int | EllipsisType | None, optional): The size of the dimension.
                Defaults to None.
            broadcastable (bool, optional): Whether this dimension can be broadcast.
                Defaults to True.
        """
        if isinstance(size, EllipsisType):
            return EllipsisDim(broadcastable)
        return super().__new__(cls)

    def __init__(self, size: int | Any | None = None, broadcastable=True):
        """
        Args:
            size: The size: a number, a HyperParameter the dim is bound to,
                or None if unknown.
            broadcastable: Whether this dimension can be broadcast.
        """
        if size is None or isinstance(size, int):
            self._size = size
        else:
            self._size = None
            self._parameters = (size,)
        self.broadcastable = broadcastable

    def __deepcopy__(self, memo) -> AxesDim:
        """A copy for another place; it is bound to the same
        HyperParameters."""
        copied = object.__new__(type(self))
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            setattr(copied, key, value if key == "_parameters" else deepcopy(value, memo))
        return copied

    def update_size(self, new_size):
        """
        Updates the size of this dimension.

        Args:
            new_size (int | None): The new size for the dimension.
        """
        self._size = new_size

    @property
    def size(self):
        """
        Returns the size of the dimension.

        Returns:
            int | None: The size of the dimension, or None if it is not set.
        """
        representative = self.representative
        if representative is not self:
            return representative.size
        # A bound HyperParameter gives the current size, e.g. after a batch
        # size was changed; the size kept from matching is the fallback.
        size = self._parameter_size()
        return size if size is not None else self._size

    @property
    def hyperparameters(self) -> tuple:
        """The HyperParameters this dim is bound to, together with those of
        every dim merged with it.

        Returns:
            tuple[HyperParameter, ...]: The bound HyperParameters; empty for
                a dim whose size is not bound to one.
        """
        return self.representative._parameters

    def _parameter_size(self) -> int | None:
        """The value of the first HyperParameter bound to this dim that has
        one."""
        return next((p.value for p in self._parameters if p.value is not None), None)

    @property
    def representative(self) -> AxesDim:
        """The dim that stands for this one and all merged into it."""
        if self.merged_into is None:
            return self
        representative = self.merged_into.representative
        self.merged_into = representative
        return representative

    def merge(self, other: AxesDim) -> AxesDim:
        """Make this dim and `other` one.

        The more informative of the two stays (an operation dim, then a dim
        of known size); the other is merged into it. An operation dim solves
        for an unknown operand once its size is known.

        Args:
            other: The dim to merge with.

        Returns:
            AxesDim: The representative of both.

        Raises:
            DataConfigMismatchError: If both sizes are known and differ.
        """
        first, second = self.representative, other.representative
        if first is second:
            return first
        keep, gone = (first, second) if first._outranks(second) else (second, first)
        if gone._feeds(keep):
            keep, gone = gone, keep
        sizes = (first.size, second.size)
        if None not in sizes and sizes[0] != sizes[1]:
            raise DataConfigMismatchError(_contradiction(first, second))
        gone.merged_into = keep
        if keep.origin is None:
            keep.origin = gone.origin
        keep._dependents = keep._dependents + gone._dependents
        keep._equal_operations = keep._equal_operations + gone._equal_operations
        keep._parameters = keep._parameters + gone._parameters
        gone._dependents, gone._equal_operations, gone._parameters = (), (), ()
        if isinstance(gone, OperationDim):
            keep._equal_operations = keep._equal_operations + (gone,)
        size = sizes[0] if sizes[0] is not None else sizes[1]
        if size is not None:
            keep._settle(size)
        return keep

    def compatible(self, other: AxesDim) -> bool:
        """Whether this dim and `other` could be merged."""
        mine, theirs = self.size, other.size
        return mine is None or theirs is None or mine == theirs

    #: Operation dims stay when merged with plain ones, as they carry
    #: relations.
    _rank = 0

    def _outranks(self, other: AxesDim) -> bool:
        """Whether this dim stays when merged with `other`: the higher rank,
        then the known size."""
        return (self._rank, self.size is not None) > (other._rank, other.size is not None)

    def _feeds(self, operation: AxesDim) -> bool:
        """Whether `operation` is computed from this dim, directly or through
        other operations; such an operation can not stand for this dim."""
        stack, seen = [self], set()
        while stack:
            for dependent in stack.pop()._dependents:
                dependent = dependent.representative
                if dependent is operation:
                    return True
                if id(dependent) not in seen:
                    seen.add(id(dependent))
                    stack.append(dependent)
        return False

    def _settle(self, size: int) -> None:
        """Take on `size` and tell everything that depends on it."""
        if self._size is None:
            self._size = size
        self._infer_parameters(size)
        for operation in self._equal_operations:
            operation._settle(size)
        for dependent in self._dependents:
            dependent.operand_changed()

    def _infer_parameters(self, size: int) -> None:
        """Tell the HyperParameters bound to this dim its size (see
        `HyperParameter.infer`)."""
        for parameter in self._parameters:
            parameter.infer(size)

    def __str__(self) -> str:
        return str(self.size)

    def __add__(self, other: AxesDim | int) -> AxesDim:
        """
        Defines addition for `AxesDim` objects, resulting in an `AddedDim`.

        Args:
            other (AxesDim): The other dimension to add.

        Returns:
            AddedDim: A new dimension representing the sum of the two.
        """
        if isinstance(other, int):
            other = AxesDim(size=other)
        return AddedDim(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other: AxesDim | int) -> AxesDim:
        """
        Defines subtraction for `AxesDim` objects, resulting in an `SubDim`.

        Args:
            other (AxesDim): The other dimension to subtract.

        Returns:
            SubDim: A new dimension representing the difference of the two.
        """
        if isinstance(other, int):
            other = AxesDim(size=other)
        return SubDim(self, other)

    def __rsub__(self, other):
        if isinstance(other, int):
            other = AxesDim(size=other)
        return SubDim(other, self)

    def __mul__(self, other: AxesDim | int) -> AxesDim:
        """
        Defines multiplication for `AxesDim` objects, resulting in an `ProductDim`.

        Args:
            other (AxesDim): The other dimension to multiply.

        Returns:
            ProductDim: A new dimension representing the product of the two.
        """
        if isinstance(other, int):
            other = AxesDim(size=other)
        return ProductDim(self, other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other: AxesDim | int) -> AxesDim:
        """
        Defines division for `AxesDim` objects, resulting in an `DivideDim`.

        Args:
            other (AxesDim): The other dimension to divide with.

        Returns:
            DivideDim: A new dimension representing the division of the two.
        """
        if isinstance(other, int):
            other = AxesDim(size=other)
        return DivideDim(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, int):
            other = AxesDim(size=other)
        return DivideDim(other, self)

class OperationDim(AxesDim):
    """
    Represents a dimension that is derived from an operation on other dimensions.
    This is for automatic referencing, to build symbolic relations.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    #: The size this dim must have, known from a merge before its operands.
    _target: int | None = None

    def __init__(self, dim_1: AxesDim, dim_2: AxesDim):
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        broadcastable = dim_1.broadcastable and dim_2.broadcastable
        super().__init__(None, broadcastable)
        for operand in self.operands:
            operand._dependents = operand._dependents + (self,)

    @property
    def operands(self) -> tuple[AxesDim, ...]:
        return (self.dim_1, self.dim_2)

    @property
    def size(self):
        representative = self.representative
        if representative is not self:
            return representative.size
        computed = self._compute()
        if computed is not None:
            return computed
        return self._target if self._target is not None else self._parameter_size()

    def _compute(self) -> int | None:
        """The size computed from the operands; None while one is unknown."""
        raise NotImplementedError

    _rank = 1

    def _settle(self, size: int) -> None:
        computed = self._compute()
        if computed is not None and computed != size:
            raise DataConfigMismatchError(
                f"{self} is {computed}, but {size} (from {self.origin}) is required."
            )
        self._target = size
        self._infer_parameters(size)
        self.solve(size)
        for operation in self._equal_operations:
            operation._settle(size)
        for dependent in self._dependents:
            dependent.operand_changed()

    def solve(self, size: int) -> None:
        """Determine an unknown operand such that this dim is `size`, where
        the answer is unique; does nothing otherwise."""

    def operand_changed(self) -> None:
        """Re-check this dim after the size of an operand became known."""
        representative = self.representative
        if representative is not self:
            computed = self._compute()
            if computed is not None and representative.size != computed:
                representative.merge(AxesDim(computed))
            return
        if self._target is not None:
            self._settle(self._target)
        elif self._compute() is not None:
            self._infer_parameters(self._compute())
            for operation in self._equal_operations:
                operation._settle(self._compute())
            for dependent in self._dependents:
                dependent.operand_changed()

    def update_size(self, new_size):
        raise RuntimeError("Can not update operation dims in place.")


class AddedDim(OperationDim):
    """
    Represents a dimension whose size is the sum of two other dimensions.
    This is for automatic referencing, to build symbolic relations.
    """

    def _compute(self) -> int | None:
        """The sum of the operands."""
        first, second = self.dim_1.size, self.dim_2.size
        return first + second if first is not None and second is not None else None

    def solve(self, size: int) -> None:
        first, second = self.dim_1.size, self.dim_2.size
        if first is not None and second is None:
            self.dim_2.merge(AxesDim(size - first))
        elif second is not None and first is None:
            self.dim_1.merge(AxesDim(size - second))

class SubDim(OperationDim):
    """
    Represents a dimension whose size is the difference of two other dimensions.
    This is for automatic referencing, to build symbolic relations.
    """

    def _compute(self) -> int | None:
        """The difference of the operands."""
        first, second = self.dim_1.size, self.dim_2.size
        return first - second if first is not None and second is not None else None

    def solve(self, size: int) -> None:
        first, second = self.dim_1.size, self.dim_2.size
        if first is not None and second is None:
            self.dim_2.merge(AxesDim(first - size))
        elif second is not None and first is None:
            self.dim_1.merge(AxesDim(size + second))

class ProductDim(OperationDim):
    """
    Represents a dimension whose size is the product of two other dimensions.
    """

    def _compute(self) -> int | None:
        """The product of the operands."""
        first, second = self.dim_1.size, self.dim_2.size
        return first * second if first is not None and second is not None else None

    def solve(self, size: int) -> None:
        first, second = self.dim_1.size, self.dim_2.size
        known, unknown = (first, self.dim_2) if second is None else (second, self.dim_1)
        if known is None or unknown.size is not None or known == 0:
            return
        if size % known != 0:
            raise DataConfigMismatchError(f"{size} is no multiple of {known} ({self}).")
        unknown.merge(AxesDim(size // known))

class DivideDim(OperationDim):
    """
    Represents a dimension whose size is the division of two other dimensions.
    """

    def _compute(self) -> int | None:
        """The first operand divided by the second, rounded down."""
        first, second = self.dim_1.size, self.dim_2.size
        return first // second if first is not None and second is not None else None

class MinimumDim(OperationDim):
    """
    Represents a dimension whose size is the minimum of two other dimensions.
    """

    def _compute(self) -> int | None:
        """The smaller of the operands."""
        first, second = self.dim_1.size, self.dim_2.size
        return min(first, second) if first is not None and second is not None else None

class SequenceProductDim(OperationDim):
    """
    A dimension whose size is the product of the sizes of a sequence of
    axes and dims, e.g. the dims merged by flattening. The sequence may hold
    an ellipsis; the size is known once the ellipsis is bound and every size
    in it is known.
    """

    def __init__(self, items: tuple):
        self.items = tuple(items)
        AxesDim.__init__(self, None)
        self._register()

    @property
    def operands(self) -> tuple:
        return self.items

    def _dims(self) -> list[AxesDim] | None:
        """The dims of the sequence; None while an ellipsis is unbound."""
        from qewton.config.sequences import expand

        dims = []
        for item in expand(self.items):
            if item.is_ellipsis:
                return None
            if isinstance(item, Axes):
                dims.extend(expand(item.shape))
            else:
                dims.append(item)
        return None if any(d.is_ellipsis for d in dims) else dims

    def _register(self) -> None:
        """Be told when a dim of the sequence or an ellipsis in it changes."""
        from qewton.config.sequences import expand

        watched = [item for item in expand(self.items) if item.is_ellipsis or not isinstance(item, Axes)]
        dims = self._dims()
        if dims is not None:
            watched += dims
        for item in watched:
            if self not in item._dependents:
                item._dependents = item._dependents + (self,)

    def _compute(self) -> int | None:
        dims = self._dims()
        if dims is None or any(d.size is None for d in dims):
            return None
        return math.prod(d.size for d in dims)

    def solve(self, size: int) -> None:
        dims = self._dims()
        if dims is None:
            return
        unknown = [d for d in dims if d.size is None]
        if len(unknown) != 1:
            return
        known = math.prod(d.size for d in dims if d.size is not None)
        if known == 0 or size % known != 0:
            raise DataConfigMismatchError(f"{size} is no multiple of {known} ({self}).")
        unknown[0].merge(AxesDim(size // known))

    def operand_changed(self) -> None:
        self._register()
        super().operand_changed()

    def __str__(self) -> str:
        return f"prod({', '.join(str(i) for i in self.items)})"


class BroadcastDim(OperationDim):
    """
    A dimension that two dims of different inputs broadcast to: the other
    one where one is 1, else both, which then must agree.
    """

    def _compute(self) -> int | None:
        sizes = (self.dim_1.size, self.dim_2.size)
        beyond_one = {size for size in sizes if size not in (None, 1)}
        if len(beyond_one) > 1:
            raise DataConfigMismatchError(f"Sizes {sizes[0]} and {sizes[1]} do not broadcast.")
        if beyond_one:
            return beyond_one.pop()
        return 1 if sizes == (1, 1) else None


class WindowDim(OperationDim):
    """
    The size of a dimension after a sliding window (a convolution or a
    pooling) with `kernel`, `stride`, `padding` and `dilation`, rounded down.
    The kernel size may be a dim itself (e.g. of a weight). Solved backwards
    only with stride 1, where the input size is unique.
    """

    def __init__(
        self,
        dim: AxesDim,
        kernel: int | AxesDim,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        self.dim = dim
        self.kernel = kernel if isinstance(kernel, AxesDim) else AxesDim(kernel)
        self.stride, self.padding, self.dilation = stride, padding, dilation
        AxesDim.__init__(self, None)
        for operand in self.operands:
            operand._dependents = operand._dependents + (self,)

    @property
    def operands(self) -> tuple[AxesDim, ...]:
        return (self.dim, self.kernel)

    def _reach(self) -> int | None:
        kernel = self.kernel.size
        return None if kernel is None else self.dilation * (kernel - 1) + 1 - 2 * self.padding

    def _compute(self) -> int | None:
        size, reach = self.dim.size, self._reach()
        return None if size is None or reach is None else (size - reach) // self.stride + 1

    def solve(self, size: int) -> None:
        reach = self._reach()
        if self.stride == 1 and self.dim.size is None and reach is not None:
            self.dim.merge(AxesDim(size - 1 + reach))


class EllipsisDim(AxesDim):
    """
    A special `AxesDim` representing an ellipsis (`...`), indicating a flexible
    number of dimensions.

    An ellipsis stands for a sequence of dims, possibly empty, until it is
    bound to one (see :mod:`qewton.config.sequences`).

    Attributes:
        bound_to (tuple[AxesDim, ...] | None): The sequence this ellipsis
            stands for; None while unknown.
        waiting (tuple): Equations of sequences to merge again once this
            ellipsis is bound.
    """

    is_ellipsis = True
    bound_to: tuple | None = None
    waiting: tuple = ()

    def compatible(self, other: AxesDim) -> bool:
        return True

    def __init__(self, broadcastable=True):
        super().__init__(size=None, broadcastable=broadcastable)

    def __str__(self) -> str:
        return "..."
