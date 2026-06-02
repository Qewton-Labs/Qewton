import unittest
from unittest.mock import patch
import tempfile
import os
import sys
import numpy as np

from pioneer.data.datasets.array_data.base import ArrayLikeDataSet
from pioneer.data.datasets.array_data.hdf5 import HDF5DataSet
from pioneer.data.datasets.array_data.numpy import NumpyDataSet
from pioneer.data.datasets.array_data.torch import TorchDataSet
from pioneer.config import (
    DataConfiguration,
    Variable,
    BatchAxes,
    FeatureAxes,
    AxesDim,
    EllipsisAxes,
    EllipsisDim,
)

try:
    import h5py
except ImportError:
    h5py = None

try:
    import torch
except ImportError:
    torch = None


class TestArrayLikeDataSet(unittest.TestCase):

    def test_init_single_data_config(self):
        data = np.random.rand(10, 5)
        config = DataConfiguration(BatchAxes(AxesDim(size=10)), FeatureAxes(shape=(5,)))
        dataset = ArrayLikeDataSet(data, config)
        self.assertEqual(dataset._data[0].shape, (10, 5))
        self.assertEqual(dataset._data_configs, [config])
        self.assertEqual(dataset._data_configs[0].axes[0].shape[0].size, 10)
        self.assertEqual(dataset._data_configs[0].axes[1].shape[0].size, 5)

    def test_init_list_data_config(self):
        data1 = np.random.rand(10, 5)
        data2 = np.random.rand(10, 3)
        config1 = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(5,)))
        config2 = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(3,)))

        dataset = ArrayLikeDataSet([data1, data2], [config1, config2])
        self.assertEqual(len(dataset._data), 2)
        self.assertEqual(dataset._data[0].shape, (10, 5))
        self.assertEqual(dataset._data[1].shape, (10, 3))

    def test_init_list_data_config_wrong_length(self):
        data = np.random.rand(10, 5)
        config1 = DataConfiguration.empty()
        config2 = DataConfiguration.empty()
        with self.assertRaisesRegex(
            AssertionError, "A separate DataConfig is required for each data object."
        ):
            ArrayLikeDataSet([data], [config1, config2])

    def test_update_configs_ellipsis_axes_error(self):
        data = np.random.rand(10)
        config = DataConfiguration(EllipsisAxes())
        with self.assertRaisesRegex(ValueError, "Ellipsis not supported at this stage."):
            ArrayLikeDataSet(data, config)

    def test_update_configs_ellipsis_dim_error(self):
        data = np.random.rand(10)
        config = DataConfiguration(FeatureAxes(shape=(EllipsisDim(),)))
        with self.assertRaisesRegex(ValueError, "Ellipsis not supported at this stage."):
            ArrayLikeDataSet(data, config)

    def test_update_configs_too_many_axes_error(self):
        data = np.random.rand(5)
        config = DataConfiguration(BatchAxes(5), FeatureAxes(shape=(5,)))
        with self.assertRaisesRegex(
            ValueError, "Too many axes in data configuration for the given data."
        ):
            ArrayLikeDataSet(data, config)

    def test_update_configs_too_few_axes_error(self):
        data = np.random.rand(10, 5)
        config = DataConfiguration(BatchAxes(10))
        with self.assertRaisesRegex(
            ValueError, "Too few axes in data configuration for the given data."
        ):
            ArrayLikeDataSet(data, config)

    def test_len(self):
        data = np.random.rand(10, 5)
        config = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(5,)))
        dataset = ArrayLikeDataSet(data, config)
        self.assertEqual(len(dataset), 10)

    def test_len_empty_data(self):
        # Create an instance with no data to test the empty case
        dataset = ArrayLikeDataSet(
            np.empty((0, 5)),
            DataConfiguration(BatchAxes(0), FeatureAxes(shape=(5,))),
        )
        self.assertEqual(len(dataset), 0)

    def test_getitem(self):
        data = np.arange(10)
        config = DataConfiguration(BatchAxes(10))
        dataset = ArrayLikeDataSet(data, config)
        self.assertEqual(dataset[0], [0])
        np.testing.assert_array_equal(dataset[1:3], [np.array([1, 2])])

    def test_data_configs_property(self):
        data = np.random.rand(10)
        config = DataConfiguration(BatchAxes(10))
        dataset = ArrayLikeDataSet(data, config)
        self.assertEqual(dataset.data_configs, [config])

    def test_get_batch(self):
        data = np.arange(10)
        config = DataConfiguration(BatchAxes(10))
        dataset = ArrayLikeDataSet(data, config)
        np.testing.assert_array_equal(dataset.get_batch([0, 2]), [np.array([0, 2])])

    def test_get_continuous_batch(self):
        data = np.arange(10)
        config = DataConfiguration(BatchAxes(10))
        dataset = ArrayLikeDataSet(data, config)
        np.testing.assert_array_equal(
            dataset.get_continuous_batch(1, 4), [np.array([1, 2, 3])]
        )

    def test_load_complete_data_no_args(self):
        data = np.random.rand(10, 5)
        config = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(5,)))
        dataset = ArrayLikeDataSet(data, config)
        np.testing.assert_array_equal(dataset.load_complete_data(), [data])

    def test_load_complete_data_data_item(self):
        data1 = np.random.rand(10, 5)
        data2 = np.random.rand(10, 3)
        config1 = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(5,)))
        config2 = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(3,)))
        dataset = ArrayLikeDataSet([data1, data2], [config1, config2])
        np.testing.assert_array_equal(dataset.load_complete_data(data_item=1), data2)

    def test_load_complete_data_variable(self):
        # Using Variable to define named segments in features
        V1 = Variable("v1", 2)
        V2 = Variable("v2", 3)
        data = np.random.rand(10, 5)
        config = DataConfiguration(BatchAxes(10), FeatureAxes(V1 + V2))
        dataset = ArrayLikeDataSet(data, config)

        v1_data = dataset.load_complete_data(variable=V1)
        np.testing.assert_array_equal(v1_data, data[:, :2])

        v2_data = dataset.load_complete_data(variable=V2)
        np.testing.assert_array_equal(v2_data, data[:, 2:])

    def test_load_complete_data_variable_not_found(self):
        data = np.random.rand(10, 5)
        config = DataConfiguration(BatchAxes(10), FeatureAxes(Variable("v1", 5)))
        dataset = ArrayLikeDataSet(data, config)
        with self.assertRaisesRegex(
            ValueError, "No such variable unknown_var in the data."
        ):
            dataset.load_complete_data(variable="unknown_var")


@unittest.skipIf(h5py is None, "h5py is not installed")
class TestHDF5DataSet(unittest.TestCase):
    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        self.tmp_path = self.tmp_file.name
        self.tmp_file.close()
        self.data = np.random.rand(10, 5)
        with h5py.File(self.tmp_path, "w") as f:
            ds = f.create_dataset("my_data", data=self.data)
            ds.attrs["version"] = "1.0"
        self.config = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(5,)))

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_init(self):
        f = h5py.File(self.tmp_path, "r")
        dataset = HDF5DataSet(f["my_data"], f, self.config)
        self.assertEqual(dataset._file, f)
        dataset.close()

    def test_from_file_success(self):
        dataset = HDF5DataSet.from_file(self.tmp_path, "my_data", self.config)
        self.assertIsInstance(dataset, HDF5DataSet)
        self.assertEqual(len(dataset), 10)
        dataset.close()

    @patch.dict(sys.modules, {"h5py": None})
    def test_from_file_import_error(self):
        with self.assertRaisesRegex(
            ImportError, "HDF5DenseLoader requires 'h5py'. Install via pip install h5py"
        ):
            HDF5DataSet.from_file(self.tmp_path, "my_data", self.config)

    def test_from_file_key_error(self):
        with self.assertRaisesRegex(KeyError, "Dataset 'wrong_key' not found in file."):
            HDF5DataSet.from_file(self.tmp_path, "wrong_key", self.config)

    def test_close(self):
        dataset = HDF5DataSet.from_file(self.tmp_path, "my_data", self.config)
        file_handle = dataset._file
        dataset.close()
        self.assertFalse(file_handle.id.valid)

    def test_context_manager(self):
        with HDF5DataSet.from_file(self.tmp_path, "my_data", self.config) as dataset:
            file_handle = dataset._file
            self.assertTrue(file_handle.id.valid)
        self.assertFalse(file_handle.id.valid)

    def test_del_closes_file(self):
        dataset = HDF5DataSet.from_file(self.tmp_path, "my_data", self.config)
        file_handle = dataset._file
        del dataset
        import gc

        gc.collect()
        self.assertFalse(file_handle.id.valid)

    def test_metadata_with_attrs(self):
        dataset = HDF5DataSet.from_file(self.tmp_path, "my_data", self.config)
        self.assertEqual(dataset.metadata, {"version": "1.0"})
        dataset.close()

    def test_metadata_without_attrs(self):
        with h5py.File(self.tmp_path, "a") as f:
            f.create_dataset("no_attrs", data=self.data)
        dataset = HDF5DataSet.from_file(self.tmp_path, "no_attrs", self.config)
        self.assertEqual(dataset.metadata, {})
        dataset.close()


class TestNumpyDataSet(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(10, 5)
        self.config = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(5,)))

    def test_init_success(self):
        dataset = NumpyDataSet(self.data, self.config)
        self.assertIsInstance(dataset._data[0], np.ndarray)

    def test_init_type_error(self):
        with self.assertRaisesRegex(
            TypeError, "NumpyDataContainer only handles numpy.ndarray"
        ):
            NumpyDataSet([1, 2, 3], self.config)

    def test_from_file_npy(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp.name, self.data)
            tmp_path = tmp.name
        try:
            dataset = NumpyDataSet.from_file(tmp_path, self.config)
            np.testing.assert_array_equal(dataset.load_complete_data()[0], self.data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


@unittest.skipIf(torch is None, "torch is not installed")
class TestTorchDataSet(unittest.TestCase):
    def setUp(self):
        self.data = torch.rand(10, 5)
        self.config = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(5,)))

    def test_init_success(self):
        dataset = TorchDataSet(self.data, self.config)
        self.assertTrue(torch.is_tensor(dataset._data[0]))

    def test_init_type_error(self):
        with self.assertRaisesRegex(TypeError, "TorchDataSet only handles torch.Tensor"):
            TorchDataSet(np.random.rand(10, 5), self.config)

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(self.data, tmp.name)
            tmp_path = tmp.name
        try:
            dataset = TorchDataSet.from_file(tmp_path, self.config)
            self.assertTrue(torch.equal(dataset.load_complete_data()[0], self.data))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_to_device(self):
        dataset = TorchDataSet(self.data, self.config)
        # Just test moving to CPU to ensure the .to() logic runs
        dataset.to("cpu")
        self.assertEqual(dataset._data[0].device.type, "cpu")

    @patch("torch.load")
    def test_from_file_import_error_logic(self, mock_load):
        # Manually trigger the ImportError branch by mocking sys.modules
        with patch.dict(sys.modules, {"torch": None}), self.assertRaises(ImportError):
            TorchDataSet.from_file("dummy.pt", self.config)

    def test_init_import_error(self):
        # Test the ImportError branch in __init__
        with patch.dict(sys.modules, {"torch": None}):
            # Since the module is already loaded in the environment,
            # this is a bit of a synthetic check for the raise logic.
            with self.assertRaises(ImportError):
                # Force a re-import attempt inside the constructor logic
                import pioneer.data.datasets.array_data.torch as torch_mod

                importlib = __import__("importlib")
                importlib.reload(torch_mod)
                torch_mod.TorchDataSet(self.data, self.config)
