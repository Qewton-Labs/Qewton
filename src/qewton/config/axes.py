from __future__ import annotations

import copy
import math
from types import EllipsisType
from typing import Any

from qewton.config.variables import Variable
from qewton.config.errors import DataConfigMismatchError
from qewton.config.saving.saving import Serializable


class Axes(Serializable):
    """
    Represents a collection of axes (dimensions) for data, including their
    sizes and types.

    An `Axes` of this base class is an axes of unknown kind; matching it
    with a `BatchAxes`, `GeometryAxes` or `FeatureAxes` gives that kind. An
    axes is a declaration: matching (see
    :class:`~qewton.config.matching.Matching`) never changes it.
    """

    #: Whether this stands for a sequence of axes (an ellipsis).
    is_ellipsis = False
    #: Whether the kind of this axes (batch, geometry, features) is known.
    kind_known = False
    #: HyperParameters bound to this object; axes have none of their own.
    parameters: tuple = ()

    @property
    def is_open(self) -> bool:
        """Whether this axes can become more concrete at a place of a graph
        (see :class:`~qewton.config.matching.Matching`): its kind, its
        content (e.g. a Variable) or one of its dims is not known yet."""
        return not self.kind_known or self._content_open() or any(
            dim.is_open for dim in self._shape
        )

    def _content_open(self) -> bool:
        """Whether the content of this axes beyond its kind can become more
        concrete."""
        return False

    @property
    def parts(self) -> tuple:
        """The dims of this axes."""
        return self._shape

    def with_parts(self, parts: tuple) -> Axes:
        """A new axes like this one with the dims `parts`, e.g. the
        stand-in of this axes at a place."""
        new = copy.copy(self)
        new._shape = tuple(parts)
        return new

    def depends_on(self, sizes) -> tuple:
        """The items whose sizes this object is computed from; none for an
        axes."""
        return ()

    def size_in(self, sizes) -> None:
        """An axes has no size of its own; its dims have."""
        return None

    @staticmethod
    def new_unknown() -> Axes:
        """A new axes of unknown kind with one unknown dim."""
        return Axes(AxesDim())

    def kind_compatible(self, other: Axes) -> bool:
        """Whether this axes and `other` can be of one kind."""
        return not self.kind_known or not other.kind_known or type(self) is type(other)

    def contents_agree(self, other: Axes) -> bool:
        """Whether the contents (e.g. Variables) of this axes and `other`
        agree; axes of different kinds (one of them unknown) have nothing to
        compare."""
        return type(self) is not type(other) or self._same_kind_contents_agree(other)

    def _same_kind_contents_agree(self, other: Axes) -> bool:
        return True

    def more_concrete_than(self, other: Axes) -> bool:
        """Whether this axes says more than `other`: a known kind, then a
        Variable or a geometry."""
        if self.kind_known and not other.kind_known:
            return True
        return type(self) is type(other) and self._content_rank(other) > 0

    def _content_rank(self, other: Axes) -> int:
        """Positive if this axes knows more beyond its kind than `other` of
        the same kind, negative if less."""
        return 0

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

    kind_known = True


class GeometryAxes(Axes):
    """
    Represents geometry-specific axes, typically used for spatial dimensions.

    It encapsulates a `Geometry` object to manage the underlying geometric shape,
    which can also be used for plotting etc.
    """

    kind_known = True

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

    def _content_open(self) -> bool:
        """A placeholder can get a geometry, a continuous geometry a
        discretization."""
        return self._is_placeholder() or not self._geometry.is_discrete

    def _content_rank(self, other: GeometryAxes) -> int:
        """A geometry knows more than a placeholder, a discretization more
        than the geometry it discretizes."""
        if self._is_placeholder() != other._is_placeholder():
            return 1 if other._is_placeholder() else -1
        if self._is_placeholder() or self._geometry is other._geometry:
            return 0
        if self._geometry.is_discretization_of(other._geometry):
            return 1
        if other._geometry.is_discretization_of(self._geometry):
            return -1
        return 0

    def _same_kind_contents_agree(self, other: GeometryAxes) -> bool:
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

    kind_known = True

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

    def _content_open(self) -> bool:
        """Features without a Variable can get one."""
        return self._variable is None

    def _content_rank(self, other: FeatureAxes) -> int:
        """A Variable knows more than none."""
        return (self._variable is not None) - (other._variable is not None)

    def _same_kind_contents_agree(self, other: FeatureAxes) -> bool:
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

    An ellipsis stands for a sequence of axes, possibly empty; matching binds
    it to one at every place (see :class:`~qewton.config.matching.Matching`).
    """

    is_ellipsis = True

    @property
    def is_open(self) -> bool:
        return True

    def __init__(self):
        super().__init__(EllipsisDim())

    def __str__(self) -> str:
        return "..."


class AxesDim(Serializable):
    """
    Represents a single dimension within a set of axes, with an optional size and
    broadcastability.

    A dim is a declaration: matching (see
    :class:`~qewton.config.matching.Matching`) never changes it. A dim of
    unknown size is *open*: at every place of a graph it stands for its own
    unknown.

    A dim may be bound to a HyperParameter, `AxesDim(node.in_neurons)`: its
    size is the current value of the HyperParameter, also after it changed
    (e.g. a batch size). It is the same at every place; an open
    HyperParameter (state INFERRED, value None) is inferred by matching.

    Attributes:
        parameters (tuple[HyperParameter, ...]): The HyperParameters this
            dim is bound to.
    """

    #: Whether this stands for a sequence of dims (an ellipsis).
    is_ellipsis = False
    #: Whether the size is computed from other dims (an operation dim).
    is_relation = False
    parameters: tuple = ()

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
            self.parameters = (size,)
        self.broadcastable = broadcastable

    def update_size(self, new_size):
        """
        Updates the size of this dimension.

        Args:
            new_size (int | None): The new size for the dimension.
        """
        self._size = new_size

    @property
    def size(self):
        """The declared size: the value of a bound HyperParameter, else the
        given number; None if unknown."""
        size = next((p.value for p in self.parameters if p.value is not None), None)
        return size if size is not None else self._size

    @property
    def is_open(self) -> bool:
        """Whether this dim can have another size at every place: it has no
        size and is bound to no HyperParameter."""
        return self._size is None and not self.parameters

    @property
    def parts(self) -> tuple:
        """The dims this one is computed from; none for a plain dim."""
        return ()

    def with_parts(self, parts: tuple) -> AxesDim:
        """A new dim like this one computed from `parts`, e.g. the stand-in
        of this dim at a place."""
        return AxesDim(None, self.broadcastable)

    def size_in(self, sizes) -> int | None:
        """The size of this dim where the sizes of other dims are given by
        `sizes` (:class:`DeclaredSizes` or a matching): the declared one."""
        return self.size

    @staticmethod
    def new_unknown() -> AxesDim:
        """A new dim of unknown size."""
        return AxesDim()

    def depends_on(self, sizes) -> tuple:
        """The items whose sizes this dim is computed from, as expanded by
        `sizes`; none for a plain dim."""
        return ()

    def solve(self, size: int, sizes) -> list[tuple[AxesDim, int]]:
        """The sizes of unknown dims this one is computed from such that it
        is `size`, where the answer is unique: (dim, size) pairs, possibly
        none."""
        return []

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

class DeclaredSizes:
    """Sizes and sequences as declared: what operation dims are computed
    from outside of a matching. A
    :class:`~qewton.config.matching.Matching` answers the same questions
    with what it found."""

    @staticmethod
    def size(dim: AxesDim) -> int | None:
        """The declared size of `dim`."""
        return dim.size

    @staticmethod
    def expand(items) -> list:
        """`items` as declared."""
        return list(items)


class OperationDim(AxesDim):
    """
    A dimension whose size is computed from other dimensions (its operands),
    e.g. a sum. It is a relation: :meth:`compute` and :meth:`solve` evaluate
    it with given sizes, as declared or as known to a matching.
    """

    is_relation = True

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, dim_1: AxesDim, dim_2: AxesDim):
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        broadcastable = dim_1.broadcastable and dim_2.broadcastable
        super().__init__(None, broadcastable)

    @property
    def operands(self) -> tuple[AxesDim, ...]:
        return (self.dim_1, self.dim_2)

    @property
    def parts(self) -> tuple:
        return self.operands

    def with_parts(self, parts: tuple) -> OperationDim:
        new = copy.copy(self)
        new.dim_1, new.dim_2 = parts
        return new

    @property
    def is_open(self) -> bool:
        return any(part.is_open for part in self.parts)

    @property
    def size(self):
        """The size computed from the declared sizes of the operands."""
        return self.compute(DeclaredSizes)

    def size_in(self, sizes) -> int | None:
        return self.compute(sizes)

    def depends_on(self, sizes) -> tuple:
        return self.operands

    def compute(self, sizes) -> int | None:
        """The size from the sizes of the operands given by `sizes` (an
        object with `size(dim)` and `expand(items)`: :class:`DeclaredSizes`
        or a matching); None while one is unknown."""
        known = (sizes.size(self.dim_1), sizes.size(self.dim_2))
        return None if None in known else self._combine(*known)

    def _combine(self, first: int, second: int) -> int:
        raise NotImplementedError

    def __str__(self) -> str:
        size = self.size
        return str(size) if size is not None else f"{type(self).__name__}"


class AddedDim(OperationDim):
    """A dimension whose size is the sum of two other dimensions."""

    def _combine(self, first: int, second: int) -> int:
        return first + second

    def solve(self, size: int, sizes) -> list[tuple[AxesDim, int]]:
        first, second = sizes.size(self.dim_1), sizes.size(self.dim_2)
        if first is not None and second is None:
            return [(self.dim_2, size - first)]
        if second is not None and first is None:
            return [(self.dim_1, size - second)]
        return []


class SubDim(OperationDim):
    """A dimension whose size is the difference of two other dimensions."""

    def _combine(self, first: int, second: int) -> int:
        return first - second

    def solve(self, size: int, sizes) -> list[tuple[AxesDim, int]]:
        first, second = sizes.size(self.dim_1), sizes.size(self.dim_2)
        if first is not None and second is None:
            return [(self.dim_2, first - size)]
        if second is not None and first is None:
            return [(self.dim_1, size + second)]
        return []


class ProductDim(OperationDim):
    """A dimension whose size is the product of two other dimensions."""

    def _combine(self, first: int, second: int) -> int:
        return first * second

    def solve(self, size: int, sizes) -> list[tuple[AxesDim, int]]:
        first, second = sizes.size(self.dim_1), sizes.size(self.dim_2)
        if (first is None) == (second is None):
            return []
        known_size, unknown = (first, self.dim_2) if second is None else (second, self.dim_1)
        if known_size == 0:
            return []
        if size % known_size != 0:
            raise DataConfigMismatchError(f"{size} is no multiple of {known_size} ({self}).")
        return [(unknown, size // known_size)]


class DivideDim(OperationDim):
    """A dimension whose size is the first of two dimensions divided by the
    second, rounded down."""

    def _combine(self, first: int, second: int) -> int:
        return first // second


class MinimumDim(OperationDim):
    """A dimension whose size is the minimum of two other dimensions."""

    def _combine(self, first: int, second: int) -> int:
        return min(first, second)


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

    @property
    def operands(self) -> tuple:
        return self.items

    def with_parts(self, parts: tuple) -> SequenceProductDim:
        new = copy.copy(self)
        new.items = tuple(parts)
        return new

    def depends_on(self, sizes) -> tuple:
        """The ellipses and dims of the sequence: its size changes when an
        ellipsis is bound or a size becomes known."""
        items = sizes.expand(self.items)
        watched = [item for item in items if item.is_ellipsis or not isinstance(item, Axes)]
        return tuple(watched + (self.dims(sizes) or []))

    def dims(self, sizes) -> list[AxesDim] | None:
        """The dims of the sequence as expanded by `sizes`; None while an
        ellipsis in it is unbound."""
        dims = []
        for item in sizes.expand(self.items):
            if item.is_ellipsis:
                return None
            if isinstance(item, Axes):
                dims.extend(sizes.expand(item.shape))
            else:
                dims.append(item)
        return None if any(d.is_ellipsis for d in dims) else dims

    def compute(self, sizes) -> int | None:
        dims = self.dims(sizes)
        if dims is None:
            return None
        sizes = [sizes.size(d) for d in dims]
        return None if None in sizes else math.prod(sizes)

    def solve(self, size: int, sizes) -> list[tuple[AxesDim, int]]:
        dims = self.dims(sizes)
        if dims is None:
            return []
        unknown = [d for d in dims if sizes.size(d) is None]
        if len(unknown) != 1:
            return []
        known_size = math.prod(sizes.size(d) for d in dims if sizes.size(d) is not None)
        if known_size == 0 or size % known_size != 0:
            raise DataConfigMismatchError(f"{size} is no multiple of {known_size} ({self}).")
        return [(unknown[0], size // known_size)]

    def __str__(self) -> str:
        return f"prod({', '.join(str(i) for i in self.items)})"


class BroadcastDim(OperationDim):
    """
    A dimension that two dims of different inputs broadcast to: the other
    one where one is 1, else both, which then must agree.
    """

    def compute(self, sizes) -> int | None:
        sizes = (sizes.size(self.dim_1), sizes.size(self.dim_2))
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

    @property
    def operands(self) -> tuple[AxesDim, ...]:
        return (self.dim, self.kernel)

    def with_parts(self, parts: tuple) -> WindowDim:
        new = copy.copy(self)
        new.dim, new.kernel = parts
        return new

    def _reach(self, sizes) -> int | None:
        kernel = sizes.size(self.kernel)
        return None if kernel is None else self.dilation * (kernel - 1) + 1 - 2 * self.padding

    def compute(self, sizes) -> int | None:
        size, reach = sizes.size(self.dim), self._reach(sizes)
        return None if size is None or reach is None else (size - reach) // self.stride + 1

    def solve(self, size: int, sizes) -> list[tuple[AxesDim, int]]:
        reach = self._reach(sizes)
        if self.stride == 1 and sizes.size(self.dim) is None and reach is not None:
            return [(self.dim, size - 1 + reach)]
        return []


class EllipsisDim(AxesDim):
    """
    A special `AxesDim` representing an ellipsis (`...`), indicating a flexible
    number of dimensions.

    An ellipsis stands for a sequence of dims, possibly empty; matching binds
    it to one at every place (see :class:`~qewton.config.matching.Matching`).
    """

    is_ellipsis = True

    @property
    def is_open(self) -> bool:
        return True

    def with_parts(self, parts: tuple) -> EllipsisDim:
        return EllipsisDim(self.broadcastable)

    def __init__(self, broadcastable=True):
        super().__init__(size=None, broadcastable=broadcastable)

    def __str__(self) -> str:
        return "..."
