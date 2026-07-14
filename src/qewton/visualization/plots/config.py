import numpy as np

from qewton.config.axes import Axes, EllipsisAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable


class PlotAxis:
    def __init__(self, n_dimensions, variable_or_axes: Variable | Axes) -> None:
        self.n_dimensions = n_dimensions
        self.variable_or_axes = variable_or_axes

    @property
    def name(self):
        if isinstance(self.variable_or_axes, Variable):
            return self.variable_or_axes.name
        return str(self.variable_or_axes)

    def get_slice(self, data_config: DataConfiguration):
        try:
            axis_idx, slc = self._find_axis_idx(self.variable_or_axes, data_config.axes)
        except ValueError:
            try:
                reverse_axis_idx, slc = self._find_axis_idx(
                    self.variable_or_axes, data_config.axes[::-1]
                )
                axis_idx = -1 - reverse_axis_idx
            except ValueError as exc:
                raise ValueError(f"Axis {self.variable_or_axes} not found in data \
                        config {data_config}.") from exc
        return axis_idx, slc

    def _find_axis_idx(self, variable_or_axis, axes) -> tuple[int, slice | None]:
        for i, i_axis in enumerate(axes):
            if i_axis is variable_or_axis:
                return i, None
            if isinstance(variable_or_axis, Variable):
                if isinstance(i_axis, (FeatureAxes, GeometryAxes)):
                    i_var = i_axis.variables
                    if variable_or_axis in i_var:
                        return i, i_var.get_slice(variable_or_axis)

            if isinstance(i_axis, EllipsisAxes):
                raise ValueError
        raise ValueError


class XAxis(PlotAxis):
    def __init__(self, variable_or_axes, log_scale=False) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.log_scale = log_scale


class YAxis(PlotAxis):
    def __init__(self, variable_or_axes, log_scale=False) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.log_scale = log_scale


class ZAxis(PlotAxis):
    def __init__(self, variable_or_axes, log_scale=False) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.log_scale = log_scale


class ColorAxis(PlotAxis):
    def __init__(self, variable_or_axes, cmap=None) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.cmap = cmap  # if not specified, plots should resort to the default cmap of their theme


class ControlAxis(PlotAxis):
    def __init__(self, init_state, n_dimensions, variable_or_axes) -> None:
        super().__init__(n_dimensions=n_dimensions, variable_or_axes=variable_or_axes)
        self._state = init_state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value


class SliderAxis(ControlAxis):
    def __init__(
        self,
        variable_or_axes,
        init_state,
        minimum,
        maximum,
        step=1,
        marks=None,
    ):
        super().__init__(init_state, n_dimensions=1, variable_or_axes=variable_or_axes)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.marks = marks


class FixedAxis(ControlAxis):
    # selects just one fixed index
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        raise ValueError("Cannot set state of FixedAxis. It is fixed.")


class TimeAxis(PlotAxis):
    # used in animations
    def __init__(self, variable_or_axes) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)


class PlotConfiguration:
    def __init__(
        self,
        axes: list[PlotAxis],
        data_config: None | DataConfiguration = None,
    ) -> None:
        self.axes = axes
        self.data_config = data_config

    def evaluate_data(
        self,
        data: np.ndarray,
        required_axis_order: list[type[PlotAxis]],
    ):
        if self.data_config is None:
            raise ValueError(
                "A data_config is required on the PlotConfiguration to evaluate data."
            )

        # resolve every axis to its real position in `data`, based on where its
        # variable/axes reference lives in the data config (negative indices from
        # `get_slice` already count correctly from the end, e.g. across an EllipsisAxes)
        resolved = []
        for axis in self.axes:
            axis_idx, slc = axis.get_slice(self.data_config)
            print(axis, axis_idx, slc)
            real_idx = axis_idx if axis_idx >= 0 else data.ndim + axis_idx
            resolved.append([axis, real_idx, slc])

        control_infos = sorted(
            (info for info in resolved if isinstance(info[0], ControlAxis)),
            key=lambda info: info[1],
            reverse=True,
        )
        remaining_infos = [
            info for info in resolved if not isinstance(info[0], ControlAxis)
        ]

        sliced_data = data
        # apply fixed/slider state highest-index-first, so collapsing a dimension
        # doesn't shift the indices of axes not yet processed
        for axis, real_idx, _ in control_infos:
            indexer = [slice(None)] * sliced_data.ndim
            indexer[real_idx] = axis.state
            sliced_data = sliced_data[tuple(indexer)]
            for other in remaining_infos:
                if other[1] > real_idx:
                    other[1] -= 1

        # slice out the relevant part of any axis shared with other variables
        # (e.g. a single variable inside a FeatureAxes or GeometryAxes)
        for axis, real_idx, slc in remaining_infos:
            if slc is not None:
                indexer = [slice(None)] * sliced_data.ndim
                indexer[real_idx] = slc
                sliced_data = sliced_data[tuple(indexer)]

        # move axes into the order required by the renderer
        axis_by_type = {type(axis): real_idx for axis, real_idx, _ in remaining_infos}
        permutation = [axis_by_type[axis_type] for axis_type in required_axis_order]
        print(axis_by_type)
        print(permutation)
        return np.transpose(sliced_data, permutation)

    def get_axis(self, axis_type):
        for axis in self.axes:
            if isinstance(axis, axis_type):
                return axis

        return None


PlotConfig = PlotConfiguration
