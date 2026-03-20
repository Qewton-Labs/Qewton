from __future__ import annotations
from types import EllipsisType
from typing import Mapping, Sequence
import copy

from .axis import Axis, FeatureAxis, BatchAxis
from .variables import Variable

# define a special data config mismatch error
class DataConfigMismatchError(ValueError):
    pass

class DataConfiguration:
    def __init__(
        self,
        dtype_units: list[DTypeUnit],
    ):
        """
        Example:
        self.axes = [batch_axis, object_axis, feature_axis]
        self.dtypes = [(list, 1), (dict, 1), (torch.tensor, (2, 5))]
        """
        self.dtype_units = dtype_units
        # self.feature_axis  # to check there is only one feature axis

    @classmethod
    def from_data(cls, data, config) -> DataConfiguration:
        raise NotImplementedError(
            "TODO: implement this method to automatically infer configuration from data"
        )

    @property
    def shape(self):
        return tuple(axis.shape for axis in [dtype.axes for dtype in self.dtype_units])

    def _get_axes(self, axis_type):
        c = 0
        out = []
        for dtype_unit in self.dtype_units:
            for axis in dtype_unit.axes:
                if isinstance(axis, axis_type):
                    out.append((c, axis))
                c += len(axis.shape)
        return out

    def __str__(
        self,
    ):  # nice and comprehensive string representation of the configuration
        type_ls = []
        for dtype in self.dtype_units:
            axes_ls = []
            for axis in dtype.axes:
                axes_ls.append([axis.name, axis.shape])
            type_ls.append([dtype.dtype, axes_ls])
        return str(type_ls)

    @property
    def batch_axes(self):
        return self._get_axes(BatchAxes)

    @property
    def feature_axis(self):
        axis = self._get_axes(FeatureAxes)
        assert (
            len(axis) == 1
        ), "There should be exactly one feature axis in the configuration."
        return axis[0]

    @property
    def geometry_axes(self):
        return self._get_axes(GeometryAxes)

    def specify_dtype(self, implementation):
        # TODO
        pass

    def unify(self, other_config: DataConfiguration) -> DataConfiguration:
        

    def fits(self, other_config: DataConfiguration) -> bool:
        """Checks if two modules can be conneted, this does not necessarily mean
        that one is a subconfig of the other, since different axes might be specified
        or unspecified."""
        try:
            self.unify(other_config)
            return True
        except DataConfigMismatchError:
            return False



class DTypeUnit:
    def __init__(self, dtype, axes):
        self.dtype = dtype
        self.axes = axes


# three general types of axes
class Axes:
    @property
    def shape(self):
        return ...

    @property
    def name(self):
        return "Axes"
    
    def shape_fits(self, other_axes, broadcast_singleton=False) -> bool:
        try:
            self.unify_shapes(self.shape, other_axes.shape, broadcast_singleton)
            return True
        except DataConfigMismatchError:
            return False
    
    @classmethod
    def unify_shapes(cls, shape1, shape2, broadcast_singleton=False):
        matching_end = []
        matching_start = []
        remaining_middle1 = []
        remaining_middle2 = []
        for s1, s2 in zip(reversed(shape1), reversed(shape2)):
            try:
                unified_dim = cls.unify_dim(s1, s2, broadcast_singleton)
                matching_end.append(unified_dim)
            except DataConfigMismatchError:
                break
        matching_end.reverse()
        if len(matching_end) == len(shape1) and len(matching_end) == len(shape2):
            #  fully matched
            return matching_end
        for s1, s2 in zip(shape1, shape2):
            try:
                unified_dim = cls.unify_dim(s1, s2, broadcast_singleton)
                matching_start.append(unified_dim)
            except DataConfigMismatchError:
                break
        remaining_middle1 = shape1[len(matching_start) : len(shape1) - len(matching_end)]
        remaining_middle2 = shape2[len(matching_start) : len(shape2) - len(matching_end)]
        # now, the middle has to contain ellipsis at its start/end to be compatible,
        # TODO
        
        return matching_start + matched_middle + matching_end
        
        
    
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
            elif dim2 == 1:
                return dim1
        raise DataConfigMismatchError(f"Cannot unify dimensions {dim1} and {dim2}.")
    
    @classmethod
    def _split_ellipsis(cls, shape):
        if Ellipsis not in shape:
            return shape, False, []

        # assume there is only one ellipsis
        i = shape.index(Ellipsis)
        if shape.count(Ellipsis) > 1:
            raise ValueError("Shape can only contain one ellipsis.")
        # return splitted version
        return shape[:i], True, shape[i+1:]


class BatchAxes(Axes):
    def __init__(self, shape=(None,)):
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def name(self):
        return "BatchAxes"

class GeometryAxes(Axes):
    def __init__(self, geometry):
        self.geometry = geometry

    @property
    def shape(self):
        return self.geometry.shape  # might be flattened?

    @property
    def name(self):
        return "GeometryAxes"


class FeatureAxes(Axes):
    def __init__(self, variables: Variable | EllipsisType):
        self.variables = variables

    @property
    def shape(self):
        return self.variables.dim

    @property
    def name(self):
        return "FeatureAxes"


# class DataConfiguration:
#     """
#     sets the basic type (numpy array, torch tensor etc) and shape of the data,
#     and also collections of these will be used to check compatibility of the algorithms
#     also include variables and their names?

#     -> later implement several configuration conversion methods (and visualization),
#     it should be possible to this during the execution of an algorithm as well as offline
#     ->  also suggest automatic conversion methods between compatible configurations

#     TODO: how to handle dictionaries, lists etc... nested structures?
#     -> Best to do this in the dataset class? Since here we only specify the general
#     shape of the data (axis.size == None, means variable size along that axis).

#     """

#     def __init__(
#         self,
#         dtype,
#         axes: list[Axis | EllipsisType],
#         feature_axis: FeatureAxis | EllipsisType,
#         connection_to_axes: Mapping[Variable, Sequence[Axis]] | None = None,
#     ):
#         assert (
#             feature_axis is ... or feature_axis in axes
#         ), "Feature axis must be one of the axes."
#         self.dtype = dtype  # TODO: Currently None if type does not matter?
#         self.axes = axes
#         self.feature_axis = feature_axis
#         self.connection_to_axes = (
#             dict(connection_to_axes) if connection_to_axes is not None else {}
#         )

#         self._batch_axis_idx: int | None = None
#         self._feature_axis_idx: int | None = None

#     @property
#     def batch_axis_idx(self) -> int:
#         if self._batch_axis_idx is not None:
#             return self._batch_axis_idx
#         self._batch_axis_idx = self._search_axis(BatchAxis)
#         return self._batch_axis_idx

#     @property
#     def feature_axis_idx(self) -> int:
#         if self._feature_axis_idx is not None:
#             return self._feature_axis_idx
#         self._feature_axis_idx = self._search_axis(FeatureAxis)
#         return self._feature_axis_idx

#     def _search_axis(self, axis_type):
#         ellipsis_seen = False
#         for i, axis in enumerate(self.axes):
#             if isinstance(axis, EllipsisType):
#                 ellipsis_seen = True
#                 break
#             if isinstance(axis, axis_type):
#                 return i
#         # check if we find axis backwards:
#         for i, axis in enumerate(reversed(self.axes)):
#             if isinstance(axis, axis_type):
#                 return -(i + 1)

#         if ellipsis_seen:
#             raise RuntimeError(
#                 "Can not find index for configurations containing ellipsis!"
#             )
#         raise ValueError(f"Data configuration has no {axis_type}.")

#     def fits(self, other_config: DataConfiguration) -> bool:
#         """Checks if another data configuration is compatible with this one.
#         Meaning that the other configuration could be a specialization of this one,
#         where some ellipsis are replaced by concrete axes or where the variables
#         in the feature axis have been reduced.
#         """
#         idx_self = 0
#         idx_other = 0
#         ellipsis_at_end = False
#         while idx_self < len(self.axes) and idx_other < len(other_config.axes):
#             if self.axes[idx_self] is ...:
#                 # Skip ellipsis
#                 idx_self += 1
#                 if idx_self == len(self.axes):
#                     # Trailing ellipsis matches everything remaining
#                     ellipsis_at_end = True
#                     break

#                 # Advance other_config.axes until we find the next self.axes element
#                 while (
#                     idx_other < len(other_config.axes)
#                     and other_config.axes[idx_other] != self.axes[idx_self]
#                 ):
#                     idx_other += 1
#             else:
#                 if other_config.axes[idx_other] != self.axes[idx_self]:
#                     return False
#                 idx_self += 1
#                 idx_other += 1

#         # Consume remaining ellipsis in self.axes
#         if not ellipsis_at_end:
#             while idx_self < len(self.axes) and (
#                 self.axes[idx_self] is ... or idx_self == len(self.axes) - 1
#             ):
#                 idx_self += 1

#             if not (idx_self == len(self.axes) and idx_other == len(other_config.axes)):
#                 return False

#         # Check if variables in feature axis are compatible (or subset)
#         if (
#             other_config.feature_axis is ...
#             or other_config.feature_axis.variables is None
#             or self.feature_axis is ...
#             or self.feature_axis.variables is None
#         ):
#             return True
#         return other_config.feature_axis.variables in self.feature_axis.variables

#     def __getitem__(self, key: int | slice | Variable) -> DataConfiguration:
#         """Slice the configuration by axis index/indices or by Variables,
#         to quickly obtain a new configuration.
#         """
#         if isinstance(key, Variable):
#             if self.feature_axis is ... or self.feature_axis.variables is None:
#                 raise ValueError(
#                     "Cannot slice by Variable when feature_axis is Ellipsis or "
#                     "has no variables."
#                 )
#             assert (
#                 key in self.feature_axis.variables
#             ), "Variable slice must be a subset of the feature axis variables"
#             # Create new axis with reduced variables
#             new_feature_axis = FeatureAxis(size=key.dim, variables=key)
#             new_axes = copy.deepcopy(self.axes)
#             new_axes[self.feature_axis_idx] = new_feature_axis
#             return type(self)(
#                 self.dtype, new_axes, new_feature_axis, self.connection_to_axes
#             )

#         if isinstance(key, (int, slice)):
#             raw = self.axes[key]

#             sliced_axes: list[Axis | EllipsisType]
#             if isinstance(raw, list):
#                 sliced_axes = raw
#             else:
#                 sliced_axes = [raw]

#             if len(sliced_axes) == 0:
#                 raise ValueError("Slice results in empty axes list")

#             if self.feature_axis in sliced_axes:
#                 feature_axis = self.feature_axis
#             else:
#                 feature_axis = ...

#             return type(self)(
#                 self.dtype, sliced_axes, feature_axis, self.connection_to_axes
#             )

#         raise TypeError(f"Unsupported slicing type: {type(key)}")

#     def __len__(self) -> int:
#         return len(self.axes)

#     def __eq__(self, other_config: object) -> bool:
#         if not isinstance(other_config, DataConfiguration):
#             return False
#         if len(other_config.axes) != len(self.axes):
#             return False
#         if self.dtype != other_config.dtype:
#             if self.dtype is not None and other_config.dtype is not None:
#                 return False
#         for i, other_axis in enumerate(other_config.axes):
#             if not other_axis == self.axes[i]:
#                 return False
#         return True

#     def axes_of(self, var: Variable) -> list[Axis]:
#         return list(self.connection_to_axes.get(var, []))

#     def variables_on_axis(self, axis: Axis) -> Variable | None:
#         for v, axes in self.connection_to_axes.items():
#             if axis in axes:
#                 return v
#         return None

#     def map_variable_to_axes(self, var: Variable, axes: list[Axis]):
#         for axis in axes:
#             assert axis in self.axes, "All axes must be part of the configuration."
#         self.connection_to_axes[var] = axes

#     def get_axis_indices_of_variables(self, var: Variable) -> list[int]:
#         if self.feature_axis is ... or self.feature_axis.variables is None:
#             return []
#         index_list: list[int] = []
#         counter: int = 0
#         for key, dim in self.feature_axis.variables.items():
#             if key in var:
#                 for i in range(dim):
#                     index_list.append(counter + i)
#             counter += dim
#         return index_list

#     def slice_axis(self, axis_idx: int, slice_values):
#         slices = [slice(None)] * len(self)
#         slices[axis_idx] = slice_values
#         return tuple(slices)
