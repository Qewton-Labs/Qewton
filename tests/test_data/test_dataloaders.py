import pytest
import torch

from qewton.config.axes import AxesDim, BatchAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.data.dataloaders.base import DataLoader
from qewton.data.datasets.array_data.base import ArrayLikeDataSet
from qewton.optim.base import EvaluationPhase


def _dataset(n: int, variable: Variable, fill_value: float) -> ArrayLikeDataSet:
    data = torch.full((n, variable.dim), float(fill_value))
    config = DataConfiguration(BatchAxes(AxesDim(n)), FeatureAxes(variable))
    return ArrayLikeDataSet(data, config)


class TestDataLoaderTestDataSet:
    def test_test_phase_pulls_from_test_data_set(self):
        X = Variable("x", 3)
        loader = DataLoader(
            data_set=_dataset(10, X, fill_value=1.0),
            batch_size=5,
            splitting_ratio=(1.0, 0.0, 0.0),
            test_data_set=_dataset(5, X, fill_value=9.0),
            shuffle_data=False,
        )
        loader.set_mode(EvaluationPhase.TEST)
        batch = loader.forward()
        assert torch.all(batch == 9.0)

    def test_train_phase_still_pulls_from_data_set(self):
        X = Variable("x", 3)
        loader = DataLoader(
            data_set=_dataset(10, X, fill_value=1.0),
            batch_size=5,
            splitting_ratio=(1.0, 0.0, 0.0),
            test_data_set=_dataset(5, X, fill_value=9.0),
            shuffle_data=False,
        )
        loader.set_mode(EvaluationPhase.TRAIN)
        batch = loader.forward()
        assert torch.all(batch == 1.0)

    def test_provides_data_in_test_phase_even_with_a_zero_test_ratio(self):
        X = Variable("x", 3)
        loader = DataLoader(
            data_set=_dataset(10, X, fill_value=1.0),
            batch_size=5,
            splitting_ratio=(1.0, 0.0, 0.0),
            test_data_set=_dataset(5, X, fill_value=9.0),
        )
        assert loader.provides_data_in_phase(EvaluationPhase.TEST) is True

    def test_without_test_data_set_test_phase_uses_the_splitting_ratio(self):
        X = Variable("x", 3)
        loader = DataLoader(
            data_set=_dataset(10, X, fill_value=1.0),
            batch_size=2,
            splitting_ratio=(0.6, 0.0, 0.4),
        )
        assert loader.provides_data_in_phase(EvaluationPhase.TEST) is True
        loader.set_mode(EvaluationPhase.TEST)
        batch = loader.forward()
        assert torch.all(batch == 1.0)

    def test_without_test_data_set_zero_test_ratio_provides_no_test_data(self):
        X = Variable("x", 3)
        loader = DataLoader(
            data_set=_dataset(10, X, fill_value=1.0),
            batch_size=2,
            splitting_ratio=(1.0, 0.0, 0.0),
        )
        assert loader.provides_data_in_phase(EvaluationPhase.TEST) is False

    def test_mismatched_variables_raise(self):
        X = Variable("x", 3)
        Y = Variable("y", 3)
        with pytest.raises(AssertionError, match="same variables"):
            DataLoader(
                data_set=_dataset(10, X, fill_value=1.0),
                batch_size=5,
                test_data_set=_dataset(5, Y, fill_value=9.0),
            )

    def test_test_data_set_smaller_than_batch_size_raises(self):
        X = Variable("x", 3)
        with pytest.raises(AssertionError, match="test dataset size"):
            DataLoader(
                data_set=_dataset(10, X, fill_value=1.0),
                batch_size=5,
                test_data_set=_dataset(3, X, fill_value=9.0),
            )
