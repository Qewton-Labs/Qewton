

class Axis():
    def __init__(self, size=None, name=None):
        self.size = size
        self.name = name

class BatchAxis(Axis):
    def __init__(self, name="batch"):
        super().__init__(size=None, name=name)

class SpatialAxis(Axis):
    def __init__(self, size=None, name="spatial"):
        super().__init__(size=size, name=name)

class ChannelAxis(Axis):
    def __init__(self, size=None, name="channels"):
        super().__init__(size=size, name=name)

class TimeAxis(Axis):
    def __init__(self, size=None, name="time"):
        super().__init__(size=size, name=name)


class DataConfiguration():
    """
    sets the basic type (numpy array, torch tensor etc) and shape of the data, and also collections of these
    will be used to check compatibility of the algorithms
    also include variables and their names?
    
    -> later implement several configuration conversion methods (and visualization), it should be possible to this during
    the execution of an algorithm as well as offline
    ->  also suggest automatic conversion methods between compatible configurations
    
    TODO: how to handle dictionaries, lists etc... nested structures?
    TODO: also allow for ellipsis in the axes?
    """
    def __init__(self, dtype, axes):
        self.dtype = dtype # e.g. numpy.float32, torch.int64, pandas ?
        self.axes = axes # list

    def fits(self, other_config):
        for i in range(len(other_config.axes)):
            if other_config.axes[-(i+1)] is not None and (self.axes[-(i+1)].size != other_config.axes[-(i+1)].size):
                return False
            if not isinstance(self.axes[-(i+1)], type(other_config.axes[-(i+1)])):
                raise Warning("Axis types do not match, but sizes do. This may lead to unexpected behavior.")
            if not self.axes[-(i+1)].name == other_config.axes[-(i+1)].name:
                raise Warning("Axis names do not match, but sizes and types do. This may lead to unexpected behavior.")
        return True


class ImageDataConfiguration(DataConfiguration):
    def __init__(self, dtype, height=None, width=None, channels=None):
        axes = [
            BatchAxis(),
            ChannelAxis(size=channels),
            SpatialAxis(size=height, name="height"),
            SpatialAxis(size=width, name="width")
        ]
        super().__init__(dtype, axes)

class TimeSeriesDataConfiguration(DataConfiguration):
    def __init__(self, dtype, time_steps=None, features=None):
        axes = [
            BatchAxis(),
            TimeAxis(size=time_steps),
            ...
        ]
        super().__init__(dtype, axes)