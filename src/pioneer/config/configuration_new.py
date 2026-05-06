from __future__ import annotations


class DataConfigMismatchError(ValueError):
    pass


class DataConfiguration:
    def __init__(self, *axes: Axes | EllipsisDim, dtype=None):
        self.axes = axes
        self.dtype = dtype

    def __str__(self):
        return f"DataConfig({[str(a) for a in self.axes]})"

    def unify_with(self, other: DataConfiguration) -> DataConfiguration:
        if self.dtype != other.dtype:
            raise DataConfigMismatchError(
                f"Found different data types {self.dtype} and {other.dtype}."
            )

        # First we check if they match from the end
        matching_end = self._match_axes(reversed(self.axes), reversed(other.axes))
        matching_end.reverse()
        if len(matching_end) == len(self.axes) and len(matching_end) == len(other.axes):
            #  fully matched
            return DataConfiguration(*matching_end, dtype=self.dtype)
        matching_start = self._match_axes(self.axes, other.axes)
        # Lastly we check the remaining middle part for compatibility (it has
        # to contain an ellipsis in at least one of the shapes to be compatible)
        remaining_middle1 = self.axes[
            len(matching_start) : len(self.axes) - len(matching_end)
        ]
        remaining_middle2 = other.axes[
            len(matching_start) : len(other.axes) - len(matching_end)
        ]
        if not any(isinstance(dim, EllipsisDim) for dim in remaining_middle1) and not any(
            isinstance(dim, EllipsisDim) for dim in remaining_middle2
        ):
            raise DataConfigMismatchError(
                f"Axes {self.axes} and {other.axes} do not match and can not be unified."
            )
        matching_middle = self._match_middle_axes(remaining_middle1, remaining_middle2)
        return DataConfiguration(
            *(matching_start + matching_middle + matching_end), dtype=self.dtype
        )

    @classmethod
    def _match_axes(cls, axes1, axes2) -> list:
        matching_axes = []
        for a1, a2 in zip(axes1, axes2):
            try:
                if isinstance(a1, EllipsisDim) or isinstance(a2, EllipsisDim):
                    # We can not unify ellipsis directly, since they may need to
                    # consume other axis as well.
                    break
                unified_axes = Axes.unify_with(a1, a2)
                matching_axes.append(unified_axes)
            except DataConfigMismatchError as e:
                raise DataConfigMismatchError(
                    f"Axes {axes1} and {axes2} do not match and can not be unified.\
                        Mismatch at axes {a1} and {a2}."
                ) from e
        return matching_axes

    @classmethod
    def _match_middle_axes(cls, remaining_middle1, remaining_middle2) -> list:
        # If one shape only is an ellipsis we are done:
        if len(remaining_middle1) == 1 and isinstance(remaining_middle1[0], EllipsisDim):
            return list(remaining_middle2)
        if len(remaining_middle2) == 1 and isinstance(remaining_middle2[0], EllipsisDim):
            return list(remaining_middle1)

        matching_middle = []
        # Determine which side has ellipsis at start vs end
        if isinstance(remaining_middle1[0], EllipsisDim):
            start_part, end_part = list(remaining_middle1), list(remaining_middle2)
        else:
            start_part, end_part = list(remaining_middle2), list(remaining_middle1)

        start_idx = len(start_part) - 1
        end_idx = len(end_part) - 2  # skip ellipsis
        added_end_shape = False

        while start_idx > 0 and end_idx >= 0:
            current = start_part[start_idx]
            try:
                Axes.unify_with(current, end_part[end_idx])  # check if they match
                # If we have a match, they could be the same axis:
                # check if all neighbors to the left also match
                # If yes we can just add everything together
                offset = end_idx + 1 - start_idx
                if offset >= 0:
                    try:
                        insert_axes = [
                            Axes.unify_with(start_part[i], end_part[offset - 1 + i])
                            for i in range(1, start_idx + 1)
                        ]
                        matching_middle.reverse()
                        matching_middle = (
                            end_part[: end_idx + 1 - start_idx]
                            + insert_axes
                            + matching_middle
                        )
                        start_idx = 0
                        added_end_shape = True
                        break
                    except DataConfigMismatchError:
                        pass
            except DataConfigMismatchError:
                pass

            matching_middle.append(current)
            start_idx -= 1

        if not added_end_shape:
            matching_middle.reverse()
            matching_middle = end_part[:-1] + matching_middle

        return matching_middle


class Axes:
    def __init__(self, shape: tuple[AxesDim, ...]):
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def unify_with(self: Axes, other: Axes) -> Axes:
        if not self.__class__ == other.__class__:
            raise DataConfigMismatchError(
                f"Cannot unify axes of different types: {self.__class__} \
                    and {other.__class__}."
            )
        unified_shape = self.unify_shapes(self.shape, other.shape)
        return self.__class__(shape=unified_shape)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({[str(s) for s in self.shape]})"

    @classmethod
    def unify_shapes(
        cls,
        shape1: tuple[AxesDim, ...],
        shape2: tuple[AxesDim, ...],
    ) -> tuple[AxesDim, ...]:
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
        matching_end = cls._match_shapes(reversed(shape1), reversed(shape2))
        matching_end.reverse()
        if len(matching_end) == len(shape1) and len(matching_end) == len(shape2):
            #  fully matched
            return tuple(matching_end)
        matching_start = cls._match_shapes(shape1, shape2)
        # Lastly we check the remaining middle part for compatibility (it has
        # to contain an ellipsis in at least one of the shapes to be compatible)
        remaining_middle1 = shape1[len(matching_start) : len(shape1) - len(matching_end)]
        remaining_middle2 = shape2[len(matching_start) : len(shape2) - len(matching_end)]
        if not any(isinstance(dim, EllipsisDim) for dim in remaining_middle1) and not any(
            isinstance(dim, EllipsisDim) for dim in remaining_middle2
        ):
            raise DataConfigMismatchError(
                f"Shapes {shape1} and {shape2} do not match and can not be unified."
            )
        matching_middle = cls._match_middle_shape(remaining_middle1, remaining_middle2)
        return tuple(matching_start + matching_middle + matching_end)

    @classmethod
    def _match_shapes(cls, shape1, shape2) -> list:
        matching_dims = []
        for s1, s2 in zip(shape1, shape2):
            try:
                if isinstance(s1, EllipsisDim) or isinstance(s2, EllipsisDim):
                    # We can not unify ellipsis directly, since they may need to
                    # consume other axis as well.
                    break
                unified_dim = AxesDim.unify_with(s1, s2)
                matching_dims.append(unified_dim)
            except DataConfigMismatchError as e:
                raise DataConfigMismatchError(
                    f"Shapes {shape1} and {shape2} do not match and can not be unified.\
                        Mismatch at dimensions {s1} and {s2}."
                ) from e
        return matching_dims

    @classmethod
    def _match_middle_shape(cls, remaining_middle1, remaining_middle2) -> list:
        # If one shape only is an ellipsis we are done:
        if len(remaining_middle1) == 1 and isinstance(remaining_middle1[0], EllipsisDim):
            return list(remaining_middle2)
        if len(remaining_middle2) == 1 and isinstance(remaining_middle2[0], EllipsisDim):
            return list(remaining_middle1)

        matching_middle = []
        # Determine which side has ellipsis at start vs end
        if isinstance(remaining_middle1[0], EllipsisDim):
            start_part, end_part = list(remaining_middle1), list(remaining_middle2)
        else:
            start_part, end_part = list(remaining_middle2), list(remaining_middle1)

        start_idx = len(start_part) - 1
        end_idx = len(end_part) - 2  # skip ellipsis
        added_end_shape = False

        while start_idx > 0 and end_idx >= 0:
            current = start_part[start_idx]
            try:
                AxesDim.unify_with(current, end_part[end_idx])  # check if they match
                # If we have a match, they could be the same axis:
                # check if all neighbors to the left also match
                # If yes we can just add everything together
                offset = end_idx + 1 - start_idx
                if offset >= 0:
                    try:
                        insert_dims = [
                            AxesDim.unify_with(start_part[i], end_part[offset - 1 + i])
                            for i in range(1, start_idx + 1)
                        ]
                        matching_middle.reverse()
                        matching_middle = (
                            end_part[: end_idx + 1 - start_idx]
                            + insert_dims
                            + matching_middle
                        )
                        start_idx = 0
                        added_end_shape = True
                        break
                    except DataConfigMismatchError:
                        pass
            except DataConfigMismatchError:
                pass

            matching_middle.append(current)
            start_idx -= 1

        if not added_end_shape:
            matching_middle.reverse()
            matching_middle = end_part[:-1] + matching_middle

        return matching_middle


class BatchAxes(Axes):
    pass


class GeometryAxes(Axes):
    def __init__(
        self, geometry: Geometry = None, shape: tuple[int | AxesDim, ...] = None
    ):

        if geometry is not None and shape is not None:
            raise ValueError("Only one of geometry or shape can be provided.")
        if geometry is not None:
            self._geometry = geometry
        elif shape is not None:
            self._geometry = Geometry(shape)
        else:
            raise ValueError("Either geometry or shape must be provided.")
        super().__init__(self._geometry.shape)

    def unify_with(self: Axes, other: Axes) -> Axes:
        if not isinstance(other, GeometryAxes):
            raise DataConfigMismatchError(
                f"Cannot unify axes of different types: {self.__class__} \
                    and {other.__class__}."
            )
        unified_geometry = self._geometry.unify_with(other._geometry)
        return self.__class__(unified_geometry)


class FeatureAxes(Axes):
    def __init__(
        self, variable: Variable = None, shape: tuple[AxesDim, ...] | None = None
    ):
        if variable is not None and shape is not None:
            raise ValueError("Only one of variable or shape can be provided.")
        if variable is not None:
            self._variable = variable
        elif shape is not None:
            self._variable = None
        super().__init__(shape if shape is not None else self._variable.shape)

    def unify_with(self: Axes, other: Axes) -> Axes:
        if not isinstance(other, FeatureAxes):
            raise DataConfigMismatchError(
                f"Cannot unify axes of different types: {self.__class__} \
                    and {other.__class__}."
            )
        unified_shape = self.unify_shapes(self.shape, other.shape)
        return self.__class__(shape=unified_shape)


class AxesDim:
    def __init__(self, size=None, broadcastable=True):
        self.size = size
        self.broadcastable = broadcastable
        self.graph = None

    def __str__(self) -> str:
        return str(self.size)

    def unify_with(self: AxesDim, other: AxesDim):
        if self == other:
            return self
        broadcastable = self.broadcastable and other.broadcastable
        out_size = self.unify_integer_dims(
            self.size, other.size, broadcast_singleton=broadcastable
        )
        return AxesDim(size=out_size, broadcastable=broadcastable)

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


class EllipsisDim(AxesDim):
    def __str__(self) -> str:
        return "..."


class Geometry:
    def __init__(self, shape: tuple[AxesDim, ...]):
        self.shape = shape

    def unify_with(self, other: Geometry) -> Geometry:
        unified_shape = Axes.unify_shapes(self.shape, other.shape)
        return Geometry(unified_shape)
