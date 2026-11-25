import numpy as np
import pytest
import torch

from qewton.config.axes import AxesDim, BatchAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.data.dataloaders.data_loader import DataLoader
from qewton.data.datasets.array_data.base import ArrayLikeDataSet
from qewton.optim.base import EvaluationPhase
from qewton.visualization.layout import Row
from qewton.visualization.plots.data import LinePlot


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


class TestDataLoaderPhaseBatchSizes:
    """validation_batch_size and test_batch_size, falling back to the
    training batch_size."""

    @staticmethod
    def _loader(**kwargs) -> DataLoader:
        X = Variable("x", 3)
        return DataLoader(
            data_set=_dataset(20, X, fill_value=1.0),
            batch_size=4,
            splitting_ratio=(0.5, 0.25, 0.25),
            shuffle_data=False,
            **kwargs,
        )

    def test_without_own_batch_sizes_every_phase_uses_batch_size(self):
        loader = self._loader()
        assert loader.validation_batch_size == 4
        assert loader.test_batch_size == 4
        for phase in EvaluationPhase:
            loader.set_mode(phase)
            assert loader.current_batch_size == 4

    def test_each_phase_draws_batches_of_its_own_size(self):
        loader = self._loader(validation_batch_size=5, test_batch_size=3)
        for phase, size in [
            (EvaluationPhase.TRAIN, 4),
            (EvaluationPhase.VALIDATION, 5),
            (EvaluationPhase.TEST, 3),
            (EvaluationPhase.ALWAYS, 4),
        ]:
            loader.set_mode(phase)
            assert loader.forward().shape[0] == size

    def test_only_validation_batch_size_given(self):
        loader = self._loader(validation_batch_size=5)
        assert loader.batch_size == 4
        assert loader.validation_batch_size == 5
        assert loader.test_batch_size == 4

    def test_batch_axes_follow_the_phase(self):
        loader = self._loader(validation_batch_size=5, test_batch_size=3)
        port = loader.output_ports[0]
        for phase, size in [
            (EvaluationPhase.TRAIN, 4),
            (EvaluationPhase.VALIDATION, 5),
            (EvaluationPhase.TEST, 3),
        ]:
            loader.set_mode(phase)
            assert port.data_configuration.shape[0] == size

    def test_the_model_behind_follows_the_phase(self):
        from qewton.algorithms import FCN
        from qewton.graphs.graphs import Graph

        loader = self._loader(validation_batch_size=5)
        model = FCN(in_neurons=3, hidden_neurons=2, out_neurons=1, n_hidden_layers=1)
        graph = Graph()
        graph.connect(loader.output_ports[0], model)
        graph.setup()
        for phase, size in [(EvaluationPhase.TRAIN, 4), (EvaluationPhase.VALIDATION, 5)]:
            graph.run(phase)
            assert graph.value_of(model.output_ports[0]).shape[0] == size
            assert model.output_ports[0].get_data_configuration(graph).shape[0] == size

    def test_changing_batch_size_reaches_phases_without_their_own(self):
        loader = self._loader(validation_batch_size=5)
        loader._batch_size.set_value(2)
        assert loader.validation_batch_size == 5
        assert loader.test_batch_size == 2

    def test_hyperparameters_hold_only_the_given_batch_sizes(self):
        names = [hp.name for hp in self._loader().hyperparameters]
        assert names == ["batch_size", "shuffle_data"]
        names = [
            hp.name
            for hp in self._loader(validation_batch_size=5, test_batch_size=3).hyperparameters
        ]
        assert names == ["batch_size", "validation_batch_size", "test_batch_size", "shuffle_data"]

    def test_a_validation_batch_larger_than_the_data_set_raises(self):
        with pytest.raises(AssertionError, match="larger than dataset size"):
            self._loader(validation_batch_size=21)

    def test_test_batch_size_is_checked_against_the_test_data_set(self):
        X = Variable("x", 3)
        loader = DataLoader(
            data_set=_dataset(10, X, fill_value=1.0),
            batch_size=5,
            test_data_set=_dataset(3, X, fill_value=9.0),
            test_batch_size=3,
        )
        loader.set_mode(EvaluationPhase.TEST)
        assert loader.forward().shape[0] == 3
        with pytest.raises(AssertionError, match="test dataset size"):
            DataLoader(
                data_set=_dataset(10, X, fill_value=1.0),
                batch_size=2,
                test_data_set=_dataset(3, X, fill_value=9.0),
                test_batch_size=4,
            )


def test_a_data_node_saves_and_loads_with_its_phase_batch_size(tmp_path):
    """The batch size of the current phase refers back to its node; a
    sampler (a DataLoader holds a numpy Generator and can not be saved yet)
    shows that this round trip works."""
    from qewton.config.saving.callables import load, save
    from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler
    from qewton.geometries.continuous.domains_2d.rectangle import Rectangle

    sampler = RandomUniformSampler(Rectangle(Variable("x", 2), [0.0, 0.0], 1.0, 1.0), 4)
    save(sampler, tmp_path / "sampler")
    loaded = load(tmp_path / "sampler")
    loaded.set_mode(EvaluationPhase.VALIDATION)
    assert loaded._phase_batch_size.node is loaded
    assert loaded._phase_batch_size.value == 4


class TestDataNodeVisualize:
    """DataNode.visualize() - runs the node on its own (no Graph) and
    plots every output port."""

    def test_lines_of_different_quantities_are_placed_side_by_side(self):
        F = Variable("f", 1)
        U = Variable("u", 1)
        loader = DataLoader(
            data_set=_multi_var_dataset(8, [F, U], [1.0, 2.0]),
            batch_size=4,
            splitting_ratio=(0.5, 0.5, 0.0),
            shuffle_data=False,
        )
        layout = loader.visualize()
        assert isinstance(layout, Row)
        assert len(layout.plots) == 2
        assert all(isinstance(p, LinePlot) for p in layout.plots)

    def test_a_line_and_a_scatter_are_placed_side_by_side(self):
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

    def test_all_outputs_share_one_slider_over_their_batch(self):
        from qewton.data import GridDataSet
        from qewton.visualization import SliderAxis

        F, X, U = Variable("f", 1), Variable("x", 1), Variable("u", 1)
        x = torch.linspace(0, 1, 10)[:, None]
        dataset = GridDataSet(
            data=[torch.rand(6, 10, 1), x.repeat(6, 1, 1), torch.rand(6, 10, 1)],
            feature_variables=[F, X, U], geometry_variable=X, point_grid=x,
        )
        loader = DataLoader(
            data_set=dataset, batch_size=3, splitting_ratio=(1.0, 0.0, 0.0), shuffle_data=False
        )
        layout = loader.visualize(mode=EvaluationPhase.TRAIN)
        sliders = {id(c) for p in layout.plots for c in p.controls if isinstance(c, SliderAxis)}
        assert len(layout.plots) == 3
        assert len(sliders) == 1

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
        (result,) = layout.plots[0].evaluate()
        values = result.positions[:, 1]
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

    def test_a_point_samplers_own_output_draws_as_points(self):
        """A PointSampler is a DataNode too; its output are the coordinates
        of its geometry, drawn as uncolored points."""
        from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
        from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
        from qewton.visualization import ScatterPlot

        X = Variable("x", 2)
        square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
        sampler = GridSampler(square, 20)
        layout = sampler.visualize()
        (plot,) = layout.plots
        assert isinstance(plot, ScatterPlot)
        assert plot.color_scale is None

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
