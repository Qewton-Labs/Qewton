from .configuration_base import DataConfiguration
from .axis import BatchAxis, FeatureAxis, SpatialAxis
from .variables import Variable


class ImageDataConfiguration(DataConfiguration):

    def __init__(
        self,
        dtype,
        height: int | None = None,
        width: int | None = None,
        channels: int | Variable | None = None,
    ):
        if isinstance(channels, Variable):
            channel_axis = FeatureAxis(size=channels.dim, variables=channels)
        else:
            channel_axis = FeatureAxis(size=channels)
        axes = [
            BatchAxis(),
            channel_axis,
            SpatialAxis(size=height, name="height"),
            SpatialAxis(size=width, name="width"),
        ]
        super().__init__(dtype, axes, channel_axis)
