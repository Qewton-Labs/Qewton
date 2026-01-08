from .base import DataSet

class TimeSeriesDataSet(DataSet):
    def __init__(self, data_config, data=None):
        super().__init__(data_config, data=data)