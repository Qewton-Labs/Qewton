import numpy as np


class PlotAxis:
    def __init__(self, n_dimensions, variable) -> None:
        self.n_dimensions = n_dimensions
        self.variable = variable
        if self.variable.dim is not None:
            assert self.n_dimensions == self.variable.dim

    @property
    def name(self):
        return self.variable.name


class XAxis(PlotAxis):
    def __init__(self, variable, log_scale=False) -> None:
        super().__init__(n_dimensions=1, variable=variable)
        self.log_scale = log_scale


class YAxis(PlotAxis):
    def __init__(self, variable, log_scale=False) -> None:
        super().__init__(n_dimensions=1, variable=variable)
        self.log_scale = log_scale


class ZAxis(PlotAxis):
    def __init__(self, variable, log_scale=False) -> None:
        super().__init__(n_dimensions=1, variable=variable)
        self.log_scale = log_scale


class ColorAxis(PlotAxis):
    def __init__(self, variable, cmap=None) -> None:
        super().__init__(n_dimensions=1, variable=variable)
        self.cmap = cmap  # if not specified, plots should resort to the default cmap of their theme


class ControlAxis(PlotAxis):
    def __init__(self, init_state, n_dimensions, variable) -> None:
        super().__init__(n_dimensions=n_dimensions, variable=variable)
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
        variable,
        init_state,
        minimum,
        maximum,
        step=1,
        marks=None,
    ):
        super().__init__(init_state, n_dimensions=1, variable=variable)
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
    def __init__(self, variable="time") -> None:
        super().__init__(n_dimensions=1, variable=variable)


class PlotConfiguration:
    def __init__(
        self,
        axes: None | list[PlotAxis],
        axes_mapping: None | dict[PlotAxis, list[int]] = None,
    ) -> None:
        self.axes = axes if axes is not None else []
        self.axes_mapping = axes_mapping if axes_mapping is not None else {}
        assert (
            axes is not None or axes_mapping is not None
        ), "Either axes or axes_mapping must be provided."
        if axes_mapping is None:
            idx = 0
            for axis in self.axes:
                self.axes_mapping[axis] = []
                for _ in range(axis.n_dimensions):
                    self.axes_mapping[axis].append(idx)
                    idx += 1

    def evaluate_data(
        self,
        data: np.ndarray,
        required_axis_order: list[type[PlotAxis]],
    ):
        sliced_data = data

        # axes that still exist after slicing
        remaining_axes = []

        for axis in self.axes:

            if isinstance(axis, ControlAxis):

                sliced_data = np.take(
                    sliced_data,
                    axis.state,
                    axis=self.axes_mapping[axis][0],
                )

            else:
                remaining_axes.append(axis)

        # current order of remaining axes
        current_order = [type(axis) for axis in remaining_axes]

        # determine permutation
        permutation = [
            current_order.index(axis_type) for axis_type in required_axis_order
        ]
        print(sliced_data.shape)
        print(permutation)

        sliced_data = np.transpose(
            sliced_data,
            permutation,
        )

        return sliced_data

    def get_axis(self, axis_type):
        for axis in self.axes:
            if isinstance(axis, axis_type):
                return axis

        return None


PlotConfig = PlotConfiguration
