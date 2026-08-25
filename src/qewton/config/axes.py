from __future__ import annotations
from types import EllipsisType

from qewton.config.variables import Variable
from qewton.config.errors import DataConfigMismatchError


def _match_remainder(inner_type, start_part, end_part, ellipsis_type):
    """Matches the remaining middle dimensions between two shapes containing ellipsis.

    This method handles the complex case where one shape has an ellipsis at the start
    (representing flexible dimensions at the beginning) and another has an ellipsis at
    the end (representing flexible dimensions at the end). It attempts to unify the
    dimensions by working backwards from the end, trying to find compatible mappings. The
    algorithm starts from the last dimension before the ellipsis in `start_part` and the
    second-to-last dimension in `end_part` (skipping the ellipsis), and works backwards.
    When a match is found, it checks if all preceding dimensions can also be unified. If
    successful, it creates a unified mapping; otherwise, it treats dimensions as separate.

    Args:
        inner_type: The type of `AxesDim` to use for unification (e.g., `AxesDim`).
        start_part (list): A list of `AxesDim` objects, where the first element is
            an `EllipsisDim`. Represents the dimensions of a shape with an ellipsis
            at the beginning.
        end_part (list): A list of `AxesDim` objects, where the last element is an
            `EllipsisDim`. Represents the dimensions of a shape with an ellipsis
            at the end.
        ellipsis_type: The class type for ellipsis dimensions (e.g., `EllipsisDim`).

    Returns:
        tuple[dict, dict]: A tuple of two dictionaries:
            - First dict: Mapping from dimensions in start_part to their unified
                            counterparts
            - Second dict: Mapping from dimensions in end_part to their unified
                            counterparts

    The returned mappings ensure that both shapes can be transformed to compatible forms
    while preserving the semantic meaning of the ellipsis dimensions. This method is
    used internally by `_match_middle_shape` to handle ellipsis unification.
    """
    matching_middle_start = {}
    matching_middle_end = {}
    start_idx = len(start_part) - 1
    end_idx = len(end_part) - 2  # skip ellipsis
    added_end_shape = False

    matching_middle_end[end_part[-1]] = []
    while start_idx > 0 and end_idx >= 0:
        current = start_part[start_idx]
        try:
            inner_type.unify_with(current, end_part[end_idx])  # check if they match
            # If we have a match, they could be the same axis:
            # check if all neighbors to the left also match
            # If yes we can just add everything together
            offset = end_idx + 1 - start_idx
            if offset >= 0:
                try:
                    insert_dims = [
                        inner_type.unify_with(start_part[i], end_part[offset - 1 + i])
                        for i in range(1, start_idx + 1)
                    ]
                    matching_middle_end[end_part[-1]].reverse()
                    matching_middle_start[start_part[0]] = end_part[:offset]
                    matching_middle_end = matching_middle_end | {
                        k: k for k in end_part[:offset]
                    }
                    for i in range(1, start_idx + 1):
                        matching_middle_start[start_part[i]] = insert_dims[i - 1][0]
                        matching_middle_end[end_part[offset + i - 1]] = insert_dims[
                            i - 1
                        ][1]
                    start_idx = 0
                    added_end_shape = True
                    break
                except DataConfigMismatchError:
                    pass
        except DataConfigMismatchError:
            pass

        matching_middle_start[current] = current
        matching_middle_end[end_part[-1]].append(current)
        start_idx -= 1

    if not added_end_shape:
        new_ellipsis = ellipsis_type()
        matching_middle_end[end_part[-1]].append(new_ellipsis)
        matching_middle_end[end_part[-1]].reverse()
        matching_middle_end = matching_middle_end | {k: k for k in end_part[:-1]}
        matching_middle_start[start_part[0]] = end_part[:-1] + [new_ellipsis]

    return matching_middle_start, matching_middle_end


class Axes:
    """
    Represents a collection of axes (dimensions) for data, including their
    sizes and types.
    """

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

    def remove_dim(self, dim):
        """
        Removes a specific dimension from the axes.

        Args:
            dim (AxesDim): The dimension to remove.
        """
        self._shape = tuple(d for d in self._shape if d != dim)

    def get_dim_idx(self, dim):
        """
        Returns the index of a specific dimension in the axes.

        Args:
            dim (AxesDim): The dimension to find.

        Returns:
            int: The index of the dimension, or None if not found.
        """
        try:
            return self._shape.index(dim)
        except ValueError:
            return None

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

    def unify_with(self: Axes, other: Axes) -> tuple[dict, dict]:
        """
        Unifies these axes with another `Axes` object, finding a common compatible shape.

        Args:
            other (Axes): The other `Axes` object to unify with.

        Returns:
            tuple[dict, dict]: A tuple of two dictionaries. The first maps dimensions
                from `self` to the unified dimensions, and the second maps dimensions
                from `other`.

        Raises:
            DataConfigMismatchError: If the axes are of different types or cannot be
                unified.
        """
        if not self.__class__ == other.__class__:
            raise DataConfigMismatchError(
                f"Cannot unify axes of different types: {self.__class__} \
                    and {other.__class__}."
            )
        return self.unify_shapes(self.shape, other.shape)

    def __str__(self) -> str:
        """
        Returns a string representation of the Axes object.

        Returns:
            str: A string representation of the Axes, showing the class name and
                the shape.
        """
        return f"{self.__class__.__name__}([{', '.join(str(s) for s in self.shape)}])"

    @classmethod
    def unify_shapes(
        cls,
        shape1: tuple[AxesDim, ...],
        shape2: tuple[AxesDim, ...],
    ) -> tuple[dict, dict]:
        """
        Unifies two shapes, `shape1` and `shape2`, to find a common compatible
        representation.

        This method attempts to match dimensions from both the start and the end
        of the shapes. It specifically handles `EllipsisDim` to allow for
        flexible matching of intermediate dimensions.

        Args:
            shape1 (tuple[AxesDim, ...]): The first shape to unify.
            shape2 (tuple[AxesDim, ...]): The second shape to unify.

        Returns:
            tuple[dict, dict]: A tuple of two dictionaries. The first maps dimensions
                from `shape1` to their unified counterparts, and the second maps
                dimensions from `shape2`.

        Raises:
            DataConfigMismatchError: If the shapes are not compatible and cannot
            be unified.

        The unification process involves:
        1. Matching dimensions from the end of both shapes.
        2. Matching dimensions from the start of both shapes.
        3. Handling any remaining middle parts, especially if they contain `EllipsisDim`.
        """
        # First we check if they match from the end
        matching_end_1, matching_end_2 = cls._match_shapes(
            reversed(shape1), reversed(shape2)
        )
        if len(matching_end_1) == len(shape1) and len(matching_end_2) == len(shape2):
            #  fully matched
            return matching_end_1, matching_end_2
        matching_start_1, matching_start_2 = cls._match_shapes(shape1, shape2)
        # Lastly we check the remaining middle part for compatibility (it has
        # to contain an ellipsis in at least one of the shapes to be compatible)
        remaining_middle1 = shape1[
            len(matching_start_1) : len(shape1) - len(matching_end_1)
        ]
        remaining_middle2 = shape2[
            len(matching_start_2) : len(shape2) - len(matching_end_2)
        ]
        if not any(isinstance(dim, EllipsisDim) for dim in remaining_middle1) and not any(
            isinstance(dim, EllipsisDim) for dim in remaining_middle2
        ):
            raise DataConfigMismatchError(
                f"Shapes {shape1} and {shape2} do not match and can not be unified."
            )
        matching_middle_1, matching_middle_2 = cls._match_middle_shape(
            remaining_middle1, remaining_middle2
        )
        # return tuple(matching_start + matching_middle + matching_end)
        return (
            matching_start_1 | matching_middle_1 | matching_end_1,
            matching_start_2 | matching_middle_2 | matching_end_2,
        )

    @classmethod
    def _match_shapes(cls, shape1, shape2) -> tuple[dict, dict]:
        """
        Matches dimensions between two shapes from their respective starting points.

        This helper method iterates through the dimensions of `shape1` and `shape2`
        (or their reversed versions) and attempts to unify corresponding `AxesDim`
        objects.

        Args:
            shape1 (Iterable[AxesDim]): The first sequence of dimensions.
            shape2 (Iterable[AxesDim]): The second sequence of dimensions.

        Returns:
            tuple[dict, dict]: Dictionaries mapping original dimensions to their
                unified forms.
        """
        matching_dims_1 = {}
        matching_dims_2 = {}
        for s1, s2 in zip(shape1, shape2):
            try:
                if isinstance(s1, EllipsisDim) or isinstance(s2, EllipsisDim):
                    # We can not unify ellipsis directly, since they may need to
                    # consume other axis as well.
                    break
                unified_dim = AxesDim.unify_with(s1, s2)
                matching_dims_1[s1], matching_dims_2[s2] = unified_dim
            except DataConfigMismatchError as e:
                raise DataConfigMismatchError(
                    f"Shapes {shape1} and {shape2} do not match and can not be unified.\
                        Mismatch at dimensions {s1} and {s2}."
                ) from e
        return matching_dims_1, matching_dims_2

    @classmethod
    def _match_middle_shape(
        cls, remaining_middle1, remaining_middle2
    ) -> tuple[dict, dict]:
        """
        Matches the middle parts of two shapes, specifically handling `EllipsisDim`.

        This method is called after matching the start and end parts of shapes.

        Args:
            remaining_middle1 (tuple[AxesDim, ...]): The middle part of the first shape.
            remaining_middle2 (tuple[AxesDim, ...]): The middle part of the second shape.

        Returns:
            tuple[dict, dict]: Dictionaries mapping original middle dimensions to their
                unified forms.
        """
        # If one shape only is an ellipsis we are done:
        if len(remaining_middle1) == 1 and isinstance(remaining_middle1[0], EllipsisDim):
            return {remaining_middle1[0]: remaining_middle2}, {
                k: k for k in remaining_middle2
            }
        if len(remaining_middle2) == 1 and isinstance(remaining_middle2[0], EllipsisDim):
            return {k: k for k in remaining_middle1}, {
                remaining_middle2[0]: remaining_middle1
            }

        # Determine which side has ellipsis at start vs end
        if isinstance(remaining_middle1[0], EllipsisDim):
            middle1, middle2 = _match_remainder(
                AxesDim, list(remaining_middle1), list(remaining_middle2), EllipsisDim
            )
        else:
            middle2, middle1 = _match_remainder(
                AxesDim, list(remaining_middle2), list(remaining_middle1), EllipsisDim
            )

        return middle1, middle2

    def update_axes(
        self, new_axes_dict: dict[AxesDim, AxesDim | tuple[AxesDim, ...]]
    ) -> bool:
        """
        Updates the dimensions of these axes based on a provided dictionary of new
        dimensions.
        This method allows for dynamic modification of the axes, including replacing
        ellipsis dimensions with concrete sequences of dimensions.

        Args:
            new_axes_dict (dict[AxesDim, AxesDim | tuple[AxesDim, ...]]): A dictionary
                mapping existing `AxesDim` objects to their new forms.

        Returns:
            bool: True if any axes were changed, False otherwise.
        """
        changed_axes = False
        new_shape = []
        for dim in self.shape:
            if dim not in new_axes_dict:
                new_shape.append(dim)
                continue
            new_dim = new_axes_dict[dim]
            if isinstance(dim, EllipsisDim):
                if isinstance(new_dim, EllipsisDim):
                    new_shape.append(dim)
                elif isinstance(new_dim, tuple | list):
                    # Here the ellipsis is replaced by a concrete new axis dim.
                    new_shape.extend(list(new_dim))
                    changed_axes = True
                    if len(new_dim) == 1:
                        changed_axes = not isinstance(new_dim[0], EllipsisDim)
                elif isinstance(new_dim, AxesDim):
                    # new_dim is not ellipsis but previously we where -> just save
                    new_shape.append(new_dim)
            elif isinstance(dim, AxesDim) and isinstance(new_dim, AxesDim):
                # Here we have to just update the inner axis dimensions, which
                # depends on the specific implementation of the axis dimension
                did_update_dim = dim.update_dim(new_dim)
                new_shape.append(dim)  # updated in place
                changed_axes = changed_axes or did_update_dim
            else:
                raise RuntimeError(
                    f"Got {new_dim} as an input, but expected AxesDim or EllipsisDim."
                )
        self._shape = tuple(new_shape)
        return changed_axes


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

    def unify_with(self: GeometryAxes, other: Axes) -> tuple[dict, dict]:
        """
        Unifies these `GeometryAxes` with another `Axes` object.

        It specifically checks for `GeometryAxes` type and delegates the unification
        to the underlying `Geometry` objects.

        Args:
            other (Axes): The other `Axes` object to unify with.

        Returns:
            tuple[dict, dict]: Dictionaries mapping original dimensions to unified
                dimensions.

        Raises:
            DataConfigMismatchError: If the other object is not `GeometryAxes` or
                unification fails.
        """
        if not isinstance(other, GeometryAxes):
            raise DataConfigMismatchError(
                f"Cannot unify axes of different types: {self.__class__} \
                    and {other.__class__}."
            )
        unified_geometry = self._geometry.unify_with(other.geometry)
        new_axes = GeometryAxes(unified_geometry)
        self_dict, other_dict = {}, {}
        for self_key, other_key, new_a in zip(self.shape, other.shape, new_axes.shape):
            self_dict[self_key] = new_a
            other_dict[other_key] = new_a
        return self_dict, other_dict


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

    def unify_with(self: FeatureAxes, other: Axes) -> tuple[dict, dict]:
        """
        Unifies these `FeatureAxes` with another `Axes` object.

        It specifically checks for `FeatureAxes` type and ensures compatibility
        of underlying `Variable` objects if they exist.

        Args:
            other (Axes): The other `Axes` object to unify with.

        Returns:
            tuple[dict, dict]: Dictionaries mapping original dimensions to unified
                dimensions.

        Raises:
            DataConfigMismatchError: If the other object is not `FeatureAxes` or
                variables do not match.
        """
        if not isinstance(other, FeatureAxes):
            raise DataConfigMismatchError(
                f"Cannot unify axes of different types: {self.__class__} \
                    and {other.__class__}."
            )
        unified_shapes = self.unify_shapes(self.shape, other.shape)
        if self._variable is not None:
            self._variable = other.variables
        elif other._variable is not None:
            other._variable = self.variables
        else:
            if self.variables != other.variables:
                raise DataConfigMismatchError(
                    "Variables do not match, when matching" + f"{self} and {other}."
                )
        return unified_shapes


class EllipsisAxes(Axes):
    """
    Represents an ellipsis in axes, indicating a flexible number of dimensions.
    """

    def __init__(self):
        super().__init__(EllipsisDim())

    def __str__(self) -> str:
        return "..."


class AxesDim:
    """
    Represents a single dimension within a set of axes, with an optional size and
    broadcastability.
    """

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

    def __init__(self, size: int | None = None, broadcastable=True):
        self._size = size
        self.broadcastable = broadcastable
        self.graph = None

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
        return self._size

    def __str__(self) -> str:
        return str(self.size)

    def unify_with(self: AxesDim, other: AxesDim):
        """
        Unifies this `AxesDim` with another `AxesDim`.

        It determines a common size, considering broadcastability.

        Args:
            other (AxesDim): The other dimension to unify with.

        Returns:
            tuple[AxesDim, AxesDim]: A tuple where both elements are the unified
                `AxesDim`.
        """
        if self == other:
            return (self, other)
        broadcastable = self.broadcastable and other.broadcastable
        out_size = self.unify_integer_dims(
            self.size, other.size, broadcast_singleton=broadcastable
        )
        out_dim = AxesDim(size=out_size, broadcastable=broadcastable)
        return (out_dim, out_dim)

    @classmethod
    def unify_integer_dims(
        cls, dim1: int | None, dim2: int | None, broadcast_singleton=False
    ):
        """
        Unifies two integer dimensions, potentially allowing singleton broadcasting.

        Args:
            dim1 (int | None): The first dimension size.
            dim2 (int | None): The second dimension size.
            broadcast_singleton (bool, optional): If True, a dimension of size 1 can be
                broadcast to match another dimension's size. Defaults to False.

        Returns:
            int | None: The unified dimension size, or None if one of the inputs was None.

        Raises:
            DataConfigMismatchError: If the dimensions cannot be unified.
        """
        if dim1 == dim2:
            return dim1
        if dim1 is None:
            return dim2
        if dim2 is None:
            return dim1
        if broadcast_singleton:
            if dim1 == 1:
                return dim2
            if dim2 == 1:
                return dim1
        raise DataConfigMismatchError(f"Cannot unify dimensions {dim1} and {dim2}.")

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

    def update_dim(self, new_dim: AxesDim) -> bool:
        """
        Updates the properties (size and broadcastability) of this dimension with those
        of `new_dim`.

        Args:
            new_dim (AxesDim): The dimension whose properties will be copied.

        Returns:
            bool: True if the dimension's properties were changed, False otherwise.
        """
        if self.size == new_dim.size and self.broadcastable == new_dim.broadcastable:
            return False
        self._size = new_dim.size
        self.broadcastable = new_dim.broadcastable
        return True


class OperationDim(AxesDim):
    """
    Represents a dimension that is derived from an operation on other dimensions.
    This is for automatic referencing, to build symbolic relations.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, dim_1: AxesDim, dim_2: AxesDim):
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        broadcastable = dim_1.broadcastable and dim_2.broadcastable
        super().__init__(self.size, broadcastable)

    def update_size(self, new_size):
        raise RuntimeError("Can not update operation dims in place.")


class AddedDim(OperationDim):
    """
    Represents a dimension whose size is the sum of two other dimensions.
    This is for automatic referencing, to build symbolic relations.
    """

    @property
    def size(self):
        """
        Calculates and returns the size of this dimension, which is the sum of its
        constituent dimensions' sizes.
        """
        size = (
            self.dim_1.size + self.dim_2.size
            if self.dim_1.size is not None and self.dim_2.size is not None
            else None
        )
        return size

    def update_dim(self, new_dim: AxesDim) -> bool:
        if new_dim.size is None:
            return False
        if self.size == new_dim.size and self.broadcastable == new_dim.broadcastable:
            return False
        # Else check how to update:
        if self.dim_1.size is None and self.dim_2.size is None:
            return False  # we can not update from this side
        if self.dim_1.size is None:
            self.dim_1.update_dim(
                AxesDim(
                    new_dim.size - self.dim_2.size,  # type: ignore
                    broadcastable=self.broadcastable,
                )
            )
        else:
            self.dim_2.update_dim(
                AxesDim(
                    new_dim.size - self.dim_1.size,  # type: ignore
                    broadcastable=self.broadcastable,
                )
            )
        self.broadcastable = new_dim.broadcastable
        return True


class SubDim(OperationDim):
    """
    Represents a dimension whose size is the difference of two other dimensions.
    This is for automatic referencing, to build symbolic relations.
    """

    @property
    def size(self):
        """
        Calculates and returns the size of this dimension, which is the sum of its
        constituent dimensions' sizes.
        """
        size = (
            self.dim_1.size - self.dim_2.size
            if self.dim_1.size is not None and self.dim_2.size is not None
            else None
        )
        return size

    def update_dim(self, new_dim: AxesDim) -> bool:
        if new_dim.size is None:
            return False
        if self.size == new_dim.size and self.broadcastable == new_dim.broadcastable:
            return False
        # Else check how to update:
        if self.dim_1.size is None and self.dim_2.size is None:
            return False  # we can not update from this side
        if self.dim_1.size is None:
            self.dim_1.update_dim(
                AxesDim(
                    new_dim.size + self.dim_2.size,  # type: ignore
                    broadcastable=self.broadcastable,
                )
            )
        else:
            self.dim_2.update_dim(
                AxesDim(
                    self.dim_1.size - new_dim.size,  # type: ignore
                    broadcastable=self.broadcastable,
                )
            )
        self.broadcastable = new_dim.broadcastable
        return True


class ProductDim(OperationDim):
    """
    Represents a dimension whose size is the product of two other dimensions.
    """

    @property
    def size(self):
        """
        Calculates and returns the size of this dimension, which is the product of
        its constituent dimensions' sizes.
        """
        size = (
            self.dim_1.size * self.dim_2.size
            if self.dim_1.size is not None and self.dim_2.size is not None
            else None
        )
        return size

    def update_dim(self, new_dim: AxesDim) -> bool:
        if new_dim.size is None:
            return False
        if self.size == new_dim.size and self.broadcastable == new_dim.broadcastable:
            return False
        # Else check how to update:
        if self.dim_1.size is None and self.dim_2.size is None:
            return False  # we can not update from this side
        if self.dim_1.size is None:
            self.dim_1.update_dim(
                AxesDim(
                    new_dim.size // self.dim_2.size,  # type: ignore
                    broadcastable=self.broadcastable,
                )
            )
        else:
            self.dim_2.update_dim(
                AxesDim(
                    new_dim.size // self.dim_1.size,  # type: ignore
                    broadcastable=self.broadcastable,
                )
            )
        self.broadcastable = new_dim.broadcastable
        return True


class DivideDim(OperationDim):
    """
    Represents a dimension whose size is the division of two other dimensions.
    """

    @property
    def size(self):
        """
        Calculates and returns the size of this dimension, which is the product of
        its constituent dimensions' sizes.
        """
        size = (
            self.dim_1.size / self.dim_2.size
            if self.dim_1.size is not None and self.dim_2.size is not None
            else None
        )
        return int(size) if size is not None else None

    def update_dim(self, new_dim: AxesDim) -> bool:
        if new_dim.size is None:
            return False
        if self.size == new_dim.size and self.broadcastable == new_dim.broadcastable:
            return False
        # Else check how to update:
        if self.dim_1.size is None and self.dim_2.size is None:
            return False  # we can not update from this side
        if self.dim_1.size is None:
            self.dim_1.update_dim(
                AxesDim(
                    new_dim.size * self.dim_2.size,  # type: ignore
                    broadcastable=self.broadcastable,
                )
            )
        else:
            self.dim_2.update_dim(
                AxesDim(
                    self.dim_1.size // new_dim.size,  # type: ignore
                    broadcastable=self.broadcastable,
                )
            )
        self.broadcastable = new_dim.broadcastable
        return True


class MinimumDim(OperationDim):
    """
    Represents a dimension whose size is the minimum of two other dimensions.
    """

    @property
    def size(self):
        """
        Calculates and returns the size of this dimension, which is the minimum of
        its constituent dimensions' sizes.
        """
        size = (
            min(self.dim_1.size, self.dim_2.size)
            if self.dim_1.size is not None and self.dim_2.size is not None
            else None
        )
        return size


class EllipsisDim(AxesDim):
    """
    A special `AxesDim` representing an ellipsis (`...`), indicating a flexible
    number of dimensions.
    """

    def __init__(self, broadcastable=True):
        super().__init__(size=None, broadcastable=broadcastable)

    def __str__(self) -> str:
        return "..."
