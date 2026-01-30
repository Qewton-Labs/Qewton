from .variables import Variable


class Axis:
    def __init__(
        self,
        size: int | None = None,
        name: str | None = None,
        variables: Variable | None = None,
    ):
        self.size = size
        self._name = name
        self.variables = variables
        if size is not None and variables is not None:
            assert (
                size == variables.dim
            ), f"Size {size} does not match the dimension of variables {variables.dim}."

    @property
    def name(self) -> str | None:
        return self._name

    def __eq__(self, other_axes: object) -> bool:
        if not isinstance(other_axes, Axis):
            return False
        return (
            self.size == other_axes.size
            and isinstance(self, type(other_axes))
            and self.name == other_axes.name
            and self.variables == other_axes.variables
        )


class BatchAxis(Axis):
    def __init__(self):
        super().__init__(size=None, name="batch")


class SpatialAxis(Axis):
    def __init__(self, size=None, name="spatial"):
        super().__init__(size=size, name=name)


class FeatureAxis(Axis):
    def __init__(self, size=None, variables: Variable | None = None):
        super().__init__(size=size, name="features", variables=variables)


class TimeAxis(Axis):
    def __init__(self, size=None):
        super().__init__(size=size, name="time")
