from __future__ import annotations
from types import EllipsisType
from .variables import Variable


# define a special data config mismatch error
class DataConfigMismatchError(ValueError):
    pass


class DataConfigDtypeMismatchError(ValueError):
    pass


class DataConfiguration:
    def __init__(
        self,
        dtype_units: DTypeUnit | list[DTypeUnit],
    ):

        if not isinstance(dtype_units, list):
            dtype_units = [dtype_units]
        self.dtype_units = dtype_units
        # _ = self.feature_axis  # to check there is only one feature axis

    @classmethod
    def empty(cls):
        return cls(DTypeUnit(None, [...]))

    @classmethod
    def from_data(cls, data) -> DataConfiguration:
        raise NotImplementedError(
            "TODO: implement this method to automatically infer configuration from data"
        )

    @classmethod
    def default_for_dtype(cls, dtype) -> DataConfiguration:
        return cls(DTypeUnit(dtype, [...]))

    @property
    def shape(self):
        shape = []
        for dtype in self.dtype_units:
            for axis in dtype.axes:
                if isinstance(axis, Axes):
                    shape.append(axis.shape)
                else:
                    shape.append(axis)  # Ellipsis
        return tuple(shape)

    def _get_axes(self, axis_type):
        out = []
        for dtype_unit in self.dtype_units:
            for axis in dtype_unit.axes:
                if isinstance(axis, axis_type):
                    out.append(axis)
        return out

    def __str__(
        self,
    ):  # nice and comprehensive string representation of the configuration
        type_ls = []
        for dtype in self.dtype_units:
            axes_ls = []
            for axis in dtype.axes:
                if isinstance(axis, Axes):
                    axes_ls.append([axis.name, axis.shape])
                else:
                    axes_ls.append(("...",))
            type_ls.append([dtype.dtype, axes_ls])
        return str(type_ls)

    @property
    def batch_axes(self):
        return self._get_axes(BatchAxes)

    @property
    def feature_axis(self):
        axis = self._get_axes(FeatureAxes)
        assert (
            len(axis) <= 1
        ), "There should be at most one feature axis in the configuration."
        return axis[0] if axis else None

    @property
    def geometry_axes(self):
        return self._get_axes(GeometryAxes)

    # def specify_dtype(self, implementation):
    #     """
    #     Sets the dtype of this configuration, if it is None.
    #     """

    def unify(self, other_config: DataConfiguration) -> DataConfiguration:
        """TODO: This will not handle any mismatches in dtypes currently.
        I think this is okay, since dtype changes should happen more directly,
        and not silently in the background? But not sure...

        Also this implementation is not really clean, and still work in progress.
        """
        # TODO: Just so it runs:
        return self

        if len(self.dtype_units) != len(other_config.dtype_units):
            raise DataConfigMismatchError(
                "Configs have different kind of Datatype shapes and can not be unified"
            )
        # Prepare inner batch axis, TODO: Maybe this can be done once
        # in the initialization/ after specify_dtype as well? Not sure
        # if some strange axes can be created in unification...
        # TODO: Maybe one also to run it once without this, to check rather
        # we can merge without collapsing -> see the one test-todo in test_configs.py
        for dtype_u in (*self.dtype_units, *other_config.dtype_units):
            dtype_u.collapse_batch_axes()

        new_dtype_units: list[DTypeUnit] = []

        for self_dtype_unit, other_dtype_unit in zip(
            self.dtype_units, other_config.dtype_units
        ):
            # Check if types are the same or one of them is None (unspecified), otherwise
            # we can not unify:
            if self_dtype_unit.dtype != other_dtype_unit.dtype:
                if self_dtype_unit.dtype is None:
                    self_dtype_unit.dtype = other_dtype_unit.dtype
                elif other_dtype_unit.dtype is None:
                    other_dtype_unit.dtype = self_dtype_unit.dtype
                else:
                    raise DataConfigDtypeMismatchError(
                        f"Found different data types {self_dtype_unit.dtype} \
                        and {other_dtype_unit.dtype} at the same position."
                    )
            # In case of ellipsis this becomes more difficult
            if (
                self_dtype_unit.contains_ellipsis()
                or other_dtype_unit.contains_ellipsis()
            ):
                # TODO: This becomes a bit ugly to match then to find which axis
                # are where.
                pass
            # Otherwise we can just compare the axis elements and see if they match
            else:
                new_dtype_units.append(DTypeUnit.unify(self_dtype_unit, other_dtype_unit))
        return DataConfiguration(new_dtype_units)

    def fits(self, other_config: DataConfiguration) -> bool:
        """Checks if two modules can be connected, this does not necessarily mean
        that one is a subconfig of the other, since different axes might be specified
        or unspecified."""
        try:
            _ = self.unify(other_config)
            return True
        except DataConfigMismatchError:
            return False


class DTypeUnit:
    def __init__(self, dtype, axes: list[Axes | EllipsisType | None]):
        self.dtype = dtype
        if axes.count(...) > 1:
            raise ValueError("Axes configuration can at most contain 1 Ellipsis.")
        self.axes = axes

    def contains_ellipsis(self) -> bool:
        return ... in self.axes

    @classmethod
    def unify(cls, self_dtype_unit, other_dtype_unit):
        # TODO: Maybe move this into the DTypeUnits
        if len(self_dtype_unit.axes) != len(other_dtype_unit.axes):
            raise DataConfigMismatchError(
                f"Found a different number of axes ({len(self_dtype_unit.axes)} \
                            vs {len(other_dtype_unit.axes)}) for the data type \
                            {self_dtype_unit.dtype}"
            )

        new_axes = []
        for self_axis, other_axis in zip(self_dtype_unit.axes, other_dtype_unit.axes):
            if self_axis is None:
                new_axes.append(other_axis)
                continue
            if other_axis is None:
                new_axes.append(self_axis)
                continue
            if not isinstance(self_axis, type(other_axis)):
                raise DataConfigMismatchError(
                    f"Could not unify configurations, found incompatible \
                    axis types {type(self_axis)} and {type(other_axis)} at the same \
                    location."
                )
            # Check if axis shape can be unified
            unified_shape = ()
            try:
                unified_shape = Axes.unify_shapes(
                    self_axis.shape, other_axis.shape  # type: ignore
                )
            except DataConfigMismatchError as e:
                raise DataConfigMismatchError from e
            # Build new axis element
            if isinstance(self_axis, BatchAxes):
                new_axis = BatchAxes(unified_shape)

            elif isinstance(self_axis, FeatureAxes):
                a, b = self_axis.variables, other_axis.variables  # type: ignore
                if b == ... or b.is_empty():
                    new_var = a
                elif a == ... or a.is_empty():
                    new_var = b
                # TODO
                new_axis = FeatureAxes(a if b == ... else b if a == ... else a * b)

            else:  # GeometryAxes
                new_axis = GeometryAxes(
                    self_axis.unify_geometry(other_axis.geometry)  # type: ignore
                )
            new_axes.append(new_axis)
        return DTypeUnit(self_dtype_unit.dtype, new_axes)

    def collapse_batch_axes(self):
        """Simplifies the axes structure inside this object if possible.
        For example a axes structure like:
            Batch(10, 5, ...), Batch(..., 7), Feature(5,)
        can be shortened to:
            Batch(10, 5, 7), Feature(5,)
        """
        new_axes_list = []

        for axis in self.axes:
            if (
                isinstance(axis, BatchAxes)
                and new_axes_list
                and isinstance(new_axes_list[-1], BatchAxes)
            ):
                shape1 = new_axes_list[-1].shape
                shape2 = axis.shape
                if shape1[-1] == ... and shape2[0] == ...:
                    merged = shape1 + shape2[1:]
                else:
                    merged = shape1 + shape2

                if merged.count(...) <= 1:
                    new_axes_list[-1] = BatchAxes(merged)
                    continue

            new_axes_list.append(axis)

        self.axes = new_axes_list


# three general types of axes
class Axes:
    @property
    def shape(self) -> tuple[int | None | EllipsisType, ...]:
        return (...,)

    @property
    def name(self) -> str:
        return "Axes"

    def shape_fits(self, other_axes, broadcast_singleton=False) -> bool:
        try:
            _ = self.unify_shapes(self.shape, other_axes.shape, broadcast_singleton)
            return True
        except DataConfigMismatchError:
            return False

    @classmethod
    def unify_shapes(
        cls,
        shape1: tuple[int | None | EllipsisType, ...],
        shape2: tuple[int | None | EllipsisType, ...],
        broadcast_singleton: bool = False,
    ) -> tuple[int | None | EllipsisType, ...]:
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
        matching_end = cls._match_shapes(
            reversed(shape1), reversed(shape2), broadcast_singleton
        )
        matching_end.reverse()
        if len(matching_end) == len(shape1) and len(matching_end) == len(shape2):
            #  fully matched
            return tuple(matching_end)
        matching_start = cls._match_shapes(shape1, shape2, broadcast_singleton)
        # Lastly we check the remaining middle part for compatibility (it has
        # to contain an ellipsis in at least one of the shapes to be compatible)
        remaining_middle1 = shape1[len(matching_start) : len(shape1) - len(matching_end)]
        remaining_middle2 = shape2[len(matching_start) : len(shape2) - len(matching_end)]
        if not ... in remaining_middle1 and not ... in remaining_middle2:
            raise DataConfigMismatchError(
                f"Shapes {shape1} and {shape2} do not match and can not be unified."
            )
        matching_middle = cls._match_middle_shape(
            remaining_middle1, remaining_middle2, broadcast_singleton
        )
        return tuple(matching_start + matching_middle + matching_end)

    @classmethod
    def _match_shapes(cls, shape1, shape2, broadcast_singleton=False) -> list:
        matching_dims = []
        for s1, s2 in zip(shape1, shape2):
            try:
                if s1 == ... or s2 == ...:
                    # We can not unify ellipsis directly, since they may need to
                    # consume other axis as well.
                    break
                unified_dim = cls.unify_dim(s1, s2, broadcast_singleton)
                matching_dims.append(unified_dim)
            except DataConfigMismatchError as e:
                raise DataConfigMismatchError(
                    f"Shapes {shape1} and {shape2} do not match and can not be unified.\
                        Mismatch at dimensions {s1} and {s2}."
                ) from e
        return matching_dims

    @classmethod
    def unify_dim(cls, dim1, dim2, broadcast_singleton=False):
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

    @classmethod
    def _match_middle_shape(
        cls, remaining_middle1, remaining_middle2, broadcast_singleton=False
    ) -> list:
        # If one shape only is an ellipsis we are done:
        if len(remaining_middle1) == 1 and ... in remaining_middle1:
            return list(remaining_middle2)
        if len(remaining_middle2) == 1 and ... in remaining_middle2:
            return list(remaining_middle1)

        matching_middle = []
        # Determine which side has ellipsis at start vs end
        if remaining_middle1[0] == ...:
            start_part, end_part = list(remaining_middle1), list(remaining_middle2)
        else:
            start_part, end_part = list(remaining_middle2), list(remaining_middle1)

        start_idx = len(start_part) - 1
        end_idx = len(end_part) - 2  # skip ellipsis
        added_end_shape = False

        while start_idx > 0 and end_idx >= 0:
            current = start_part[start_idx]
            try:
                cls.unify_dim(
                    current, end_part[end_idx], broadcast_singleton
                )  # check if they match
                # If we have a match, they could be the same axis:
                # check if all neighbors to the left also match
                # If yes we can just add everything together
                offset = end_idx + 1 - start_idx
                if offset >= 0:
                    try:
                        insert_dims = [
                            cls.unify_dim(
                                start_part[i],
                                end_part[offset - 1 + i],
                                broadcast_singleton,
                            )
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

    # @classmethod
    # def _split_ellipsis(cls, shape):
    #     if Ellipsis not in shape:
    #         return shape, False, []

    #     # assume there is only one ellipsis
    #     i = shape.index(Ellipsis)
    #     if shape.count(Ellipsis) > 1:
    #         raise ValueError("Shape can only contain one ellipsis.")
    #     # return splitted version
    #     return shape[:i], True, shape[i + 1 :]


class BatchAxes(Axes):
    def __init__(self, shape=(None,)):
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def name(self) -> str:
        return "BatchAxes"


class GeometryAxes(Axes):
    # TODO: What would a convolutional layer have as a default config?
    # In 2D for example:
    # [Dtype: (..., Featureaxis(Variable), GeometryAxes(None, None))]?
    # This is currently not really possible? Do we have some dummy
    # Geometry later on for this? And in unify shapes we need to check
    # if we have such geometry?
    def __init__(self, geometry):
        self.geometry = geometry

    @property
    def shape(self):
        return self.geometry.shape  # TODO: might be flattened?

    def unify_geometry(self, other_geometry):
        # TODO: Implement the above comment?....
        pass

    @property
    def name(self) -> str:
        return "GeometryAxes"


class FeatureAxes(Axes):
    def __init__(self, variables: Variable | EllipsisType = Variable()):
        self.variables = variables

    @property
    def shape(self):
        if self.variables is ...:
            return (...,)
        if isinstance(self.variables.dim, tuple):
            return self.variables.dim
        return (self.variables.dim,)

    @property
    def name(self) -> str:
        return "FeatureAxes"
