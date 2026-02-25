import importlib.util

from .datasets.base import DataSet

from .datacontainer.base import DataContainer
from .datacontainer.array_data.base import ArrayLikeDataContainer
from .datacontainer.array_data.numpy_container import NumpyDataContainer
from .datacontainer.array_data.torch_container import TorchDataContainer
from .datacontainer.array_data.hdf5_container import HDF5DataContainer

if importlib.util.find_spec("torch") is not None:
    from .datasets.pytorch_dataset import TorchDataSet
