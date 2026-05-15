import importlib.util

from .datasets.base import DataSet

from .datasets.array_data.base import ArrayLikeDataSet
from .datasets.array_data.numpy_container import NumpyDataContainer
from .datasets.array_data.torch_container import TorchDataContainer
from .datasets.array_data.hdf5_container import HDF5DataContainer

if importlib.util.find_spec("torch") is not None:
    from .dataloaders.pytorch_dataset import TorchDataSet
