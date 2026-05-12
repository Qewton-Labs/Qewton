from __future__ import annotations
from types import EllipsisType

from .variables import Variable
from .errors import DataConfigMismatchError


def _match_remainder(inner_type, start_part, end_part, ellipsis_type):
    """Matches the remaining middle dimensions between two shapes containing ellipsis.

    This method handles the complex case where one shape has an ellipsis at the start
    (representing flexible dimensions at the beginning) and another has an ellipsis at
    the end (representing flexible dimensions at the end). It attempts to unify the
    dimensions by working backwards from the end, trying to find compatible mappings.

    The algorithm starts from the last dimension before the ellipsis in start_part and
    the second-to-last dimension in end_part (skipping the ellipsis), and works
    backwards.
    When a match is found, it checks if all preceding dimensions can also be unified.
    If successful, it creates a unified mapping; otherwise, it treats dimensions as
    separate.

    Args:
        start_part (list): List of dimensions from the shape with ellipsis at the
                            start. The first element should be an EllipsisDim.
        end_part (list): List of dimensions from the shape with ellipsis at the end.
                        The last element should be an EllipsisDim.

    Returns:
        tuple[dict, dict]: A tuple of two dictionaries:
            - First dict: Mapping from dimensions in start_part to their unified
                            counterparts
            - Second dict: Mapping from dimensions in end_part to their unified
                            counterparts

    Notes:
        This method is used internally by _match_middle_shape to handle ellipsis
        unification.
        The returned mappings ensure that both shapes can be transformed to compatible
        forms while preserving the semantic meaning of the ellipsis dimensions.
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
    def __init__(self, *shape: int | AxesDim | EllipsisType):
        new_shape = []
        for s in shape:
            if isinstance(s, int):
                new_shape.append(AxesDim(size=s))
            elif isinstance(s, EllipsisType):
                new_shape.append(EllipsisDim())
            else:
                new_shape.append(s)
        self._shape = tuple(new_shape)

    @property
    def shape(self):
        return self._shape

    def unify_with(self: Axes, other: Axes) -> tuple[dict, dict]:
        if not self.__class__ == other.__class__:
            raise DataConfigMismatchError(
                f"Cannot unify axes of different types: {self.__class__} \
                    and {other.__class__}."
            )
        return self.unify_shapes(self.shape, other.shape)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}([{', '.join(str(s) for s in self.shape)}])"

    @classmethod
    def unify_shapes(
        cls,
        shape1: tuple[AxesDim, ...],
        shape2: tuple[AxesDim, ...],
    ) -> tuple[dict, dict]:
        """Returns a shape that both ``shape1`` and ``shape2`` are compatible with.

        The resulting shape is chosen to have the smallest number of axes and
        the least ambiguity. Unknown or flexible dimensions may be represented
        by ``None`` or ``Ellipsis``.

        Args:
            shape1, shape2 (tuple):
                The shapes to unify.
            broadcast_singleton (bool, optional):
                Whether dimensions of size 1 can be broadcast to match larger
                dimensions. Defaults to False.

        Raises:
            DataConfigMismatchError:
                If the shapes are not compatible.

        Returns:
            tuple[int | None | EllipsisType, ...]:
                A shape that both input shapes can be transformed to.

        Notes:
            Examples:

            Basic matching:
                (2, 3) and (2, 3) -> (2, 3)

            Broadcasting disabled:
                (1, 3) and (2, 3) -> Error

            Broadcasting enabled:
                (1, 3) and (2, 3) -> (2, 3)

            Unknown dimensions:
                (None, 3) and (2, 3) -> (2, 3)

            Conflicting dimensions:
                (2, 3) and (4, 3) -> Error

            Using Ellipsis:
                (1, ..., 3) and (1, 2, 3) -> (1, 2, 3)

            Smallest matching shape:
                (5, ...) and (..., 5) -> (5,)
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
    pass


class GeometryAxes(Axes):
    def __init__(
        self,
        geometry: Geometry | None = None,
        shape: tuple[int | AxesDim, ...] | None = None,
    ):
        if geometry is not None and shape is not None:
            raise ValueError("Only one of geometry or shape can be provided.")
        if geometry is not None:
            self._geometry = geometry
        elif shape is not None:
            self._geometry = Geometry(shape)
        else:
            raise ValueError("Either geometry or shape must be provided.")
        super().__init__(*self._geometry.shape)

    def unify_with(self: Axes, other: Axes) -> tuple[dict, dict]:
        if not isinstance(other, GeometryAxes):
            raise DataConfigMismatchError(
                f"Cannot unify axes of different types: {self.__class__} \
                    and {other.__class__}."
            )
        unified_geometry = self._geometry.unify_with(other._geometry)
        new_axes = GeometryAxes(unified_geometry)
        self_dict, other_dict = {}, {}
        for self_key, other_key, new_a in zip(self.shape, other.shape, new_axes.shape):
            self_dict[self_key] = new_a
            other_dict[other_key] = new_a
        return self_dict, other_dict


class FeatureAxes(Axes):
    def __init__(
        self,
        variable: Variable | None = None,
        shape: tuple[int | AxesDim, ...] | None = None,
    ):
        if variable is not None and shape is not None:
            raise ValueError("Only one of variable or shape can be provided.")
        if variable is not None:
            self._variable = variable
        elif shape is not None:
            self._variable = None
        super().__init__(*(shape if shape is not None else self._variable.shape))

    def unify_with(self: Axes, other: Axes) -> tuple[dict, dict]:
        if not isinstance(other, FeatureAxes):
            raise DataConfigMismatchError(
                f"Cannot unify axes of different types: {self.__class__} \
                    and {other.__class__}."
            )
        unified_shapes = self.unify_shapes(self.shape, other.shape)
        return unified_shapes


class EllipsisAxes(Axes):
    def __init__(self):
        super().__init__(EllipsisDim())

    def __str__(self) -> str:
        return "..."


class AxesDim:
    def __new__(cls, size=None, broadcastable=True) -> AxesDim:
        if isinstance(size, EllipsisType):
            return EllipsisDim(broadcastable)
        return super().__new__(cls)

    def __init__(self, size=None, broadcastable=True):
        self._size = size
        self.broadcastable = broadcastable
        self.graph = None

    @property
    def size(self):
        return self._size

    def __str__(self) -> str:
        return str(self.size)

    def unify_with(self: AxesDim, other: AxesDim):
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

    def __add__(self, other: AxesDim) -> AxesDim:
        return AddedDim(self, other)

    def update_dim(self, new_dim: AxesDim) -> bool:
        # TODO: Override this in subclasses, e.g. in AddedDim
        if self.size == new_dim.size and self.broadcastable == new_dim.broadcastable:
            return False
        self.size = new_dim.size
        self.broadcastable = new_dim.broadcastable
        return True


class AddedDim(AxesDim):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, dim_1, dim_2):
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        broadcastable = dim_1.broadcastable and dim_2.broadcastable
        super().__init__(self.size, broadcastable)

    @property
    def size(self):
        size = (
            self.dim_1.size + self.dim_2.size
            if self.dim_1.size is not None and self.dim_2.size is not None
            else None
        )
        return size


class MinimumDim(AxesDim):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, dim_1, dim_2):
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        broadcastable = dim_1.broadcastable and dim_2.broadcastable
        super().__init__(self.size, broadcastable)

    @property
    def size(self):
        size = (
            min(self.dim_1.size, self.dim_2.size)
            if self.dim_1.size is not None and self.dim_2.size is not None
            else None
        )
        return size


class EllipsisDim(AxesDim):
    def __init__(self, broadcastable=True):
        super().__init__(size=None, broadcastable=broadcastable)

    def __str__(self) -> str:
        return "..."


class Geometry:
    def __init__(self, shape: tuple[int | AxesDim, ...]):
        self.shape = tuple(AxesDim(size=s) if isinstance(s, int) else s for s in shape)

    def unify_with(self, other: Geometry) -> Geometry:
        unified_shape = Axes.unify_shapes(self.shape, other.shape)
        return Geometry(tuple(unified_shape[1].values()))
