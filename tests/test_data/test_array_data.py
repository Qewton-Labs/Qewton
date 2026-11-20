import copy
import pickle
import unittest
from unittest.mock import patch
import tempfile
import os
import sys
import numpy as np

from qewton.backends.numpy.base import NumPyBackend
from qewton.backends.torch.base import TorchBackend
from qewton.data.datasets.array_data.base import ArrayLikeDataSet
from qewton.data.datasets.array_data.hdf5 import HDF5DataSet
from qewton.data import DataLoader
from qewton.config import (
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
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp_dir.name, "data.h5")
        self.x = np.random.rand(10, 5)
        self.y = np.random.rand(10, 1)
        with h5py.File(self.path, "w") as f:
            ds = f.create_dataset("x", data=self.x)
            ds.attrs["version"] = "1.0"
            f.create_dataset("group/y", data=self.y)
        self.x_config = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(5,)))
        self.y_config = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(1,)))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _dataset(self, backend=NumPyBackend, **kwargs):
        return HDF5DataSet(
            self.path, ["x", "group/y"], [self.x_config, self.y_config], backend, **kwargs
        )

    def test_reads_keys_into_memory(self):
        dataset = self._dataset()
        self.assertTrue(dataset.data_set_loaded)
        self.assertEqual(len(dataset), 10)
        x, y = dataset.load_complete_data()
        self.assertIsInstance(x, np.ndarray)
        np.testing.assert_allclose(x, self.x.astype(np.float32))
        np.testing.assert_allclose(y, self.y.astype(np.float32))

    def test_single_key(self):
        dataset = HDF5DataSet(self.path, "x", self.x_config, NumPyBackend)
        self.assertEqual(len(dataset.load_complete_data()), 1)

    def test_default_dtype_is_float32(self):
        x, _ = self._dataset().load_complete_data()
        self.assertEqual(x.dtype, np.float32)

    def test_dtype_none_keeps_file_dtype(self):
        x, _ = self._dataset(dtype=None).load_complete_data()
        self.assertEqual(x.dtype, np.float64)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_torch_backend(self):
        dataset = self._dataset(TorchBackend)
        x, _ = dataset.load_complete_data()
        self.assertTrue(torch.is_tensor(x))
        self.assertEqual(x.dtype, torch.float32)

    def test_shuffled_batch(self):
        batch = self._dataset().get_batch(np.array([7, 2, 2, 5]))
        np.testing.assert_allclose(batch[0], self.x[[7, 2, 2, 5]].astype(np.float32))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_in_torch_dataloader(self):
        dataset = self._dataset(TorchBackend)
        loader = DataLoader(dataset, batch_size=4, shuffle_data=True, shuffle_seed=0)
        loader.setup()
        x, y = loader.forward()
        self.assertTrue(torch.is_tensor(x))
        self.assertEqual(tuple(x.shape), (4, 5))
        self.assertEqual(tuple(y.shape), (4, 1))

    def test_load_on_setup(self):
        dataset = self._dataset(load_on_setup=True)
        self.assertFalse(dataset.data_set_loaded)
        dataset.setup()
        self.assertTrue(dataset.data_set_loaded)

    def test_file_is_not_kept_open(self):
        self._dataset()
        # Truncating fails if any handle to the file is still open.
        with h5py.File(self.path, "w"):
            pass

    def test_pickle_and_deepcopy(self):
        dataset = self._dataset()
        for copied in (pickle.loads(pickle.dumps(dataset)), copy.deepcopy(dataset)):
            np.testing.assert_array_equal(
                copied.load_complete_data(data_item=0), dataset.load_complete_data(data_item=0)
            )

    def test_reads_file_again_after_deserialization(self):
        from qewton.config.saving.callables import save, load

        dataset = self._dataset()
        save_path = os.path.join(self.tmp_dir.name, "dataset")
        save(dataset, save_path)
        loaded = load(save_path)
        self.assertIsInstance(loaded, HDF5DataSet)
        self.assertFalse(loaded.data_set_loaded)
        loaded.setup()
        np.testing.assert_array_equal(
            loaded.load_complete_data(data_item=1), dataset.load_complete_data(data_item=1)
        )

    def test_from_file(self):
        dataset = HDF5DataSet.from_file(self.path, "x", self.x_config, NumPyBackend)
        self.assertIsInstance(dataset, HDF5DataSet)
        self.assertEqual(len(dataset), 10)

    @patch.dict(sys.modules, {"h5py": None})
    def test_import_error(self):
        with self.assertRaisesRegex(ImportError, "HDF5DataSet requires 'h5py'"):
            self._dataset()

    def test_missing_key(self):
        with self.assertRaisesRegex(KeyError, "Dataset 'wrong_key' not found"):
            HDF5DataSet(self.path, "wrong_key", self.x_config, NumPyBackend)

    def test_group_key(self):
        with self.assertRaisesRegex(TypeError, "is a group"):
            HDF5DataSet(self.path, "group", self.y_config, NumPyBackend)

    def test_metadata(self):
        self.assertEqual(self._dataset().metadata, {"x": {"version": "1.0"}, "group/y": {}})


class TestNumpyDataSet(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(10, 5)
        self.config = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(5,)))

    def test_init_success(self):
        dataset = ArrayLikeDataSet(self.data, self.config, NumPyBackend)
        self.assertIsInstance(dataset._data[0], np.ndarray)

    def test_init_type_error(self):
        with self.assertRaises(TypeError):
            ArrayLikeDataSet([1, 2, 3], self.config, NumPyBackend)

    def test_from_file_npy(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp.name, self.data)
            tmp_path = tmp.name
        try:
            if os.path.exists(tmp_path):
                dataset = ArrayLikeDataSet.from_file(tmp_path, self.config, NumPyBackend)
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
        dataset = ArrayLikeDataSet(self.data, self.config, TorchBackend)
        self.assertTrue(torch.is_tensor(dataset._data[0]))

    def test_init_type_error(self):
        with self.assertRaises(TypeError):
            ArrayLikeDataSet(np.random.rand(10, 5), self.config, TorchBackend)

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(self.data, tmp.name)
            tmp_path = tmp.name
        try:
            dataset = ArrayLikeDataSet.from_file(tmp_path, self.config, TorchBackend)
            self.assertTrue(torch.equal(dataset.load_complete_data()[0], self.data))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_to_device(self):
        dataset = ArrayLikeDataSet(self.data, self.config, TorchBackend)
        # Just test moving to CPU to ensure the .to() logic runs
        dataset.to("cpu")
        self.assertEqual(dataset._data[0].device.type, "cpu")


class TestBackendHandling(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(10, 5)
        self.config = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(5,)))

    def test_backend_inferred_from_data(self):
        dataset = ArrayLikeDataSet(self.data, self.config)
        self.assertIs(dataset.backend, NumPyBackend)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_backend_inferred_from_torch_data(self):
        dataset = ArrayLikeDataSet(torch.rand(10, 5), self.config)
        self.assertIs(dataset.backend, TorchBackend)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_mixed_backends_raise(self):
        config2 = DataConfiguration(BatchAxes(10), FeatureAxes(shape=(5,)))
        with self.assertRaises(TypeError):
            ArrayLikeDataSet([self.data, torch.rand(10, 5)], [self.config, config2])

    def test_unknown_data_type_raises(self):
        with self.assertRaisesRegex(ValueError, "No known backend"):
            ArrayLikeDataSet([[1.0] * 5] * 10, self.config)

    def test_no_data_and_no_backend_raises(self):
        with self.assertRaisesRegex(ValueError, "Pass the backend explicitly"):
            ArrayLikeDataSet([None], [self.config])

    def test_from_file_load_on_setup_single_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "data.npy")
            np.save(path, self.data)
            dataset = ArrayLikeDataSet.from_file(
                path, self.config, NumPyBackend, load_on_setup=True
            )
            self.assertFalse(dataset.data_set_loaded)

            dataset.setup()

        self.assertTrue(dataset.data_set_loaded)
        np.testing.assert_array_equal(dataset.load_complete_data(data_item=0), self.data)

    def test_setup_validates_loaded_data(self):
        wrong_config = DataConfiguration(BatchAxes(10))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "data.npy")
            np.save(path, self.data)
            dataset = ArrayLikeDataSet.from_file(
                path, wrong_config, NumPyBackend, load_on_setup=True
            )
            with self.assertRaisesRegex(ValueError, "Too few axes"):
                dataset.setup()

    def test_setup_does_not_reload_loaded_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "data.npy")
            np.save(path, self.data)
            dataset = ArrayLikeDataSet.from_file(path, self.config, NumPyBackend)
        # The file is gone, so a reload would fail.
        dataset.setup()
        np.testing.assert_array_equal(dataset.load_complete_data(data_item=0), self.data)

    def test_to_keeps_unloaded_items(self):
        dataset = ArrayLikeDataSet([None], [self.config], backend=NumPyBackend)
        dataset.to("cpu")
        self.assertEqual(dataset._data, [None])

    def test_does_not_alias_input_list(self):
        data = [self.data]
        dataset = ArrayLikeDataSet(data, [self.config])
        dataset.to("cpu")
        self.assertIsNot(dataset._data, data)

    def test_reload_from_paths_after_deserialization(self):
        from qewton.config.saving.callables import save, load

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "data.npy")
            np.save(path, self.data)
            dataset = ArrayLikeDataSet.from_file(path, self.config, NumPyBackend)
            save(dataset, os.path.join(tmp_dir, "dataset"))

            loaded = load(os.path.join(tmp_dir, "dataset"))
            self.assertIs(loaded.backend, NumPyBackend)
            self.assertFalse(loaded.data_set_loaded)

            loaded.setup()
            np.testing.assert_array_equal(loaded.load_complete_data(data_item=0), self.data)

    def test_save_data_round_trip(self):
        from qewton.config.saving.callables import save, load

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = ArrayLikeDataSet(self.data, self.config, save_data=True)
            save(dataset, os.path.join(tmp_dir, "dataset"))
            loaded = load(os.path.join(tmp_dir, "dataset"))
        np.testing.assert_array_equal(loaded.load_complete_data(data_item=0), self.data)
