"""Rules of the bar, image, surface and vector plots and of combined
position bindings, beyond the test matrix."""

import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization import (
    BarPlot,
    ColorScale,
    Figure,
    ImagePlot,
    Row,
    ScatterPlot,
    SliderAxis,
    SurfacePlot,
    VectorPlot,
    VectorScale,
)
from qewton.visualization.renderers import PlotlyRenderer

from recording_renderer import RecordingRenderer

T, X, Y = Variable("t", 1), Variable("x", 2), Variable("y", 3)
U, V, RGB = Variable("u", 1), Variable("v", 2), Variable("rgb", 3)


def _grid_data(grid=None, features=U * V * RGB, steps=None, seed=0):
    grid = grid if grid is not None else GridGeometry.build_unit_grid(X, 4)
    axes = [GeometryAxes(grid), FeatureAxes(features)]
    if steps is not None:
        axes.insert(0, BatchAxes(steps))
    config = DataConfiguration(*axes)
    data = np.random.default_rng(seed).random(config.shape)
    return data, config, axes[0]


class TestPositions:
    def test_position_and_a_free_channel_combine(self):
        data, config, _ = _grid_data()
        plot = SurfacePlot(data, config, position=X, z=U)
        (result,) = plot.evaluate()
        assert result.positions.shape == (16, 3)
        np.testing.assert_allclose(result.positions[:, 2], data[..., 0].ravel())
        assert plot.axis_title("z") == "$u$"

    def test_a_channel_is_bound_once(self):
        data, config, _ = _grid_data()
        with pytest.raises(ValueError, match="binds y already"):
            ScatterPlot(data, config, position=X, y=U)

    def test_channels_are_bound_in_order(self):
        data, config, _ = _grid_data()
        with pytest.raises(ValueError, match="bind x, y and possibly z"):
            ScatterPlot(data, config, x=U, z=U)


class TestBar:
    def test_a_bar_per_batch_element(self):
        config = DataConfiguration(BatchAxes(5), FeatureAxes(U))
        data = np.arange(5.0)[:, None]
        batch = config.axes[0]
        (result,) = BarPlot(data, config, x=batch, y=U).evaluate()
        np.testing.assert_allclose(result.positions, [[i, i] for i in range(5)])

    def test_bars_are_drawn_with_plotly(self):
        config = DataConfiguration(BatchAxes(5), FeatureAxes(U))
        data = np.arange(5.0)[:, None]
        plot = BarPlot(data, config, x=config.axes[0], y=U, color=U)
        backend = Figure(plot).draw()
        assert [t.type for t in backend.data if t.type == "bar"] == ["bar"]


class TestImage:
    def test_colors_are_clipped_and_arranged_as_an_image(self):
        data, config, _ = _grid_data()
        data[..., 3:6] *= 2
        (result,) = ImagePlot(data, config, position=X, color=RGB).evaluate()
        assert result.colors.shape == (4, 4, 3)
        assert result.colors.max() <= 1.0
        assert result.color_scale is None

    def test_an_image_takes_rgb_or_rgba(self):
        data, config, _ = _grid_data()
        with pytest.raises(ValueError, match="RGB or RGBA"):
            ImagePlot(data, config, position=X, color=V)

    def test_an_image_takes_no_color_scale(self):
        data, config, _ = _grid_data()
        with pytest.raises(TypeError, match="color_scale"):
            ImagePlot(data, config, position=X, color=RGB, color_scale=ColorScale())

    def test_an_image_gives_no_colorbar(self):
        data, config, _ = _grid_data()
        plot = ImagePlot(data, config, position=X, color=RGB)
        backend = Figure(plot, renderer=RecordingRenderer()).draw()
        assert backend["colorbars"] == {}

    def test_plotly_draws_images_on_uniform_grids_only(self):
        points = np.stack(
            np.meshgrid([0.0, 0.1, 1.0], [0.0, 0.5, 1.0], indexing="ij"), axis=-1
        )
        data, config, _ = _grid_data(GridGeometry(X, points))
        plot = ImagePlot(data, config, position=X, color=RGB)
        with pytest.raises(ValueError, match="uniform x spacing"):
            Figure(plot, renderer=PlotlyRenderer()).draw()


class TestSurface:
    def test_a_volume_is_drawn_by_its_boundary(self):
        box = Box(Y, [0.0] * 3, 1.0, 1.0, 1.0).create_mesh(0.6)
        config = DataConfiguration(GeometryAxes(box), FeatureAxes(U))
        data = np.random.default_rng(0).random(config.shape)
        plot = SurfacePlot(data, config, position=Y, color=U)
        (result,) = plot.evaluate()
        np.testing.assert_array_equal(result.cells, box.boundary_cells)

    def test_edges_are_drawn_in_3d(self):
        data, config, _ = _grid_data()
        plot = SurfacePlot(data, config, position=X, z=U, show_edges=True)
        backend = Figure(plot).draw()
        assert [t.type for t in backend.data] == ["mesh3d", "scatter3d"]


class TestVector:
    def test_the_longest_arrow_has_the_scale_length(self):
        data, config, batch = _grid_data(steps=3)
        scale = VectorScale(length=0.5)
        plot = VectorPlot(
            data, config, position=X, vector=V, vector_scale=scale,
            controls=[SliderAxis(batch)],
        )
        Figure(plot, renderer=RecordingRenderer()).draw()
        longest = max(
            np.linalg.norm(data[k, ..., 1:3], axis=-1).max() for k in range(3)
        )
        assert scale.range[1] == pytest.approx(longest)
        (result,) = plot.evaluate()
        lengths = np.linalg.norm(result.vectors, axis=-1)
        np.testing.assert_allclose(
            lengths, np.linalg.norm(data[0, ..., 1:3], axis=-1).ravel() / longest * 0.5
        )

    def test_a_shared_vector_scale_spans_both_plots(self):
        (d1, c1, _), (d2, c2, _) = _grid_data(seed=1), _grid_data(seed=2)
        d2[..., 1:3] *= 10
        scale = VectorScale()
        plots = [
            VectorPlot(d, c, position=X, vector=V, vector_scale=scale)
            for d, c in ((d1, c1), (d2, c2))
        ]
        Figure(Row(*plots), renderer=RecordingRenderer()).draw()
        assert scale.range[1] == pytest.approx(np.linalg.norm(d2[..., 1:3], axis=-1).max())

    def test_normalized_arrows_have_one_length(self):
        data, config, _ = _grid_data()
        scale = VectorScale(length=0.2, normalize=True)
        (result,) = VectorPlot(
            data, config, position=X, vector=V, vector_scale=scale
        ).evaluate()
        np.testing.assert_allclose(np.linalg.norm(result.vectors, axis=-1), 0.2)

    def test_every_nth_arrow_is_drawn(self):
        data, config, _ = _grid_data()
        (result,) = VectorPlot(
            data, config, position=X, vector=V, color=V, subsample=3
        ).evaluate()
        assert len(result.positions) == len(result.vectors) == len(result.values) == 6

    def test_a_vector_has_one_component_per_position(self):
        data, config, _ = _grid_data()
        with pytest.raises(ValueError, match="3 components for 2"):
            VectorPlot(data, config, position=X, vector=RGB)

    def test_a_vector_is_a_feature(self):
        data, config, _ = _grid_data()
        with pytest.raises(ValueError, match="one feature"):
            VectorPlot(data, config, position=X, vector=X)


class TestScatterOnALine:
    def test_one_position_lies_on_the_x_axis_of_a_2d_cell(self):
        T = Variable("t", 1)
        config = DataConfiguration(
            GeometryAxes(GridGeometry.build_unit_grid(T, 5)), FeatureAxes(U)
        )
        plot = ScatterPlot(np.zeros(config.shape), config, position=T)
        assert plot.dim == 2 and plot.n_positions == 1
        (result,) = plot.evaluate()
        np.testing.assert_allclose(result.positions[:, 0], np.linspace(0, 1, 5))
        np.testing.assert_allclose(result.positions[:, 1], 0.0)
