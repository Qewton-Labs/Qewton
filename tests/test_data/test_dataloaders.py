import numpy as np
import pytest
import torch

from qewton.config.axes import AxesDim, BatchAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.data.dataloaders.base import DataLoader
from qewton.data.datasets.array_data.base import ArrayLikeDataSet
from qewton.optim.base import EvaluationPhase
from qewton.visualization.layout import Overlay, Row
from qewton.visualization.plots.data.curve import LinePlot


def _dataset(n: int, variable: Variable, fill_value: float) -> ArrayLikeDataSet:
    data = torch.full((n, variable.dim), float(fill_value))
    config = DataConfiguration(BatchAxes(AxesDim(n)), FeatureAxes(variable))
    return ArrayLikeDataSet(data, config)


def _multi_var_dataset(
    n: int, variables: list[Variable], fill_values: list[float]
) -> ArrayLikeDataSet:
    datas = [torch.full((n, v.dim), float(fv)) for v, fv in zip(variables, fill_values)]
    configs = [DataConfiguration(BatchAxes(AxesDim(n)), FeatureAxes(v)) for v in variables]
    return ArrayLikeDataSet(datas, configs)


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


class TestDataNodeVisualize:
    """DataNode.visualize() - runs the node on its own (no Graph) and
    plots every output port."""

    def test_overlays_all_curve_outputs(self):
        F = Variable("f", 1)
        U = Variable("u", 1)
        loader = DataLoader(
            data_set=_multi_var_dataset(8, [F, U], [1.0, 2.0]),
            batch_size=4,
            splitting_ratio=(0.5, 0.5, 0.0),
            shuffle_data=False,
        )
        layout = loader.visualize()
        assert isinstance(layout, Overlay)
        assert len(layout.plots) == 2
        assert all(isinstance(p, LinePlot) for p in layout.plots)

    def test_rows_mixed_curve_and_non_curve_outputs(self):
        F = Variable("f", 1)
        XY = Variable("x", 1) * Variable("y", 1)
        datas = [torch.full((8, 1), 1.0), torch.rand(8, 2)]
        configs = [
            DataConfiguration(BatchAxes(AxesDim(8)), FeatureAxes(F)),
            DataConfiguration(BatchAxes(AxesDim(8)), FeatureAxes(XY)),
        ]
        dataset = ArrayLikeDataSet(datas, configs)
        loader = DataLoader(
            data_set=dataset, batch_size=4, splitting_ratio=(0.5, 0.5, 0.0), shuffle_data=False
        )
        layout = loader.visualize()
        assert isinstance(layout, Row)
        assert len(layout.plots) == 2

    def test_respects_an_explicit_mode_without_a_graph(self):
        X = Variable("x", 1)
        loader = DataLoader(
            data_set=_dataset(8, X, fill_value=1.0),
            batch_size=4,
            splitting_ratio=(1.0, 0.0, 0.0),
            test_data_set=_dataset(4, X, fill_value=9.0),
            shuffle_data=False,
        )
        layout = loader.visualize(mode=EvaluationPhase.TEST)
        values = np.asarray(layout.plots[0].evaluate().y)
        assert np.all(values == 9.0)

    def test_defaults_to_validation_mode(self):
        X = Variable("x", 1)
        loader = DataLoader(
            data_set=_dataset(8, X, fill_value=1.0),
            batch_size=4,
            splitting_ratio=(0.5, 0.5, 0.0),
            shuffle_data=False,
        )
        loader.visualize()
        assert loader.mode == EvaluationPhase.VALIDATION

    def test_a_point_samplers_own_output_draws_as_a_point_cloud(self):
        """A PointSampler is a DataNode too - its own output IS its
        geometry's own coordinate Variable, so it draws as an uncolored
        point cloud (a Row, not an Overlay, for dim>=2) rather than
        crashing on auto_plot's usual quantity-vs-geometry dispatch."""
        from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
        from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
        from qewton.visualization.plots.data.points import PointCloudPlot

        X = Variable("x", 2)
        square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
        sampler = GridSampler(square, 20)
        layout = sampler.visualize()
        assert isinstance(layout, Row)
        assert isinstance(layout.plots[0], PointCloudPlot)
        assert layout.plots[0].color is None

    def test_a_mode_with_no_data_raises_a_clear_error_instead_of_crashing(self):
        """splitting_ratio=(1.0, 0.0, 0.0) reserves nothing for VALIDATION -
        DataLoader.forward() would return None for it, which used to
        surface as a bare 'NoneType is not subscriptable' deep inside
        Node.run() instead of pointing at the actual cause."""
        X = Variable("x", 1)
        loader = DataLoader(
            data_set=_dataset(8, X, fill_value=1.0),
            batch_size=4,
            splitting_ratio=(1.0, 0.0, 0.0),
            shuffle_data=False,
        )
        with pytest.raises(ValueError, match="provides no data"):
            loader.visualize()
