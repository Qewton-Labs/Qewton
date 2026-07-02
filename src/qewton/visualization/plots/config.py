import numpy as np


class PlotAxis:
    def __init__(self, n_dimensions, name) -> None:
        self.n_dimensions = n_dimensions
        self.name = name


class XAxis(PlotAxis):
    def __init__(self, name, log_scale=False) -> None:
        super().__init__(n_dimensions=1, name=name)
        self.log_scale = log_scale


class YAxis(PlotAxis):
    def __init__(self, name, log_scale=False) -> None:
        super().__init__(n_dimensions=1, name=name)
        self.log_scale = log_scale


class ZAxis(PlotAxis):
    def __init__(self, name, log_scale=False) -> None:
        super().__init__(n_dimensions=1, name=name)
        self.log_scale = log_scale


class ColorAxis(PlotAxis):
    def __init__(self, name, cmap=None) -> None:
        super().__init__(n_dimensions=1, name=name)
        self.cmap = cmap  # if not specified, plots should resort to the default cmap of their theme


class ControlAxis(PlotAxis):
    def __init__(self, init_state, n_dimensions, name) -> None:
        super().__init__(n_dimensions=n_dimensions, name=name)
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
        init_state,
        minimum,
        maximum,
        step=1,
        name="",
        marks=None,
    ):
        super().__init__(init_state, n_dimensions=1, name=name)
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
    def __init__(self, name="time") -> None:
        super().__init__(n_dimensions=1, name=name)


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

    def evaluate_data(self, data):
        sliced_data = data
        for axis in self.axes:
            if isinstance(axis, ControlAxis):
                sliced_data = np.take(
                    sliced_data,
                    axis.state,
                    axis=self.axes_mapping[axis][0],
                )
        return sliced_data


PlotConfig = PlotConfiguration
