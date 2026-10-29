"""Names of Variables in Plotly: one math text for MathJax, Plotly markup
where MathJax does not render (sliders, buttons, mathjax=False), plain
text in Dash widgets."""

import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization import (
    DashApplication,
    FixedAxis,
    Figure,
    HeatmapPlot,
    Selector,
    SliderAxis,
)
from qewton.visualization.renderers import PlotlyRenderer
from qewton.visualization.renderers.plotly.text import as_markup, as_math, as_plain

X = Variable("x", 2)
U, P = Variable("u", 1), Variable("p", 1)


class TestConversion:
    @pytest.mark.parametrize(
        "text, math, markup, plain",
        [
            ("$x_1$", "$x_1$", "<i>x</i><sub>1</sub>", "x_1"),
            ("$t$ = 0.5", r"$t\text{ = 0.5}$", "<i>t</i> = 0.5", "t = 0.5"),
            ("$u$, batch = 1", r"$u\text{, batch = 1}$", "<i>u</i>, batch = 1", "u, batch = 1"),
            ("$x_{12}^2$", "$x_{12}^2$", "<i>x</i><sub>12</sub><sup>2</sup>", "x_12^2"),
            (r"$\alpha$", r"$\alpha$", "<i>α</i>", "α"),
            ("batch = 1", "batch = 1", "batch = 1", "batch = 1"),
            ("a $ b", "a $ b", "a $ b", "a $ b"),
            (None, None, None, None),
        ],
    )
    def test_texts(self, text, math, markup, plain):
        assert as_math(text) == math
        assert as_markup(text) == markup
        assert as_plain(text) == plain

    def test_markup_escapes_plain_text(self):
        assert as_markup("a < b") == "a &lt; b"


def _figure(mathjax):
    batch = BatchAxes(3)
    config = DataConfiguration(
        batch, GeometryAxes(GridGeometry.build_unit_grid(X, 4)), FeatureAxes(U * P)
    )
    data = np.random.default_rng(0).random(config.shape)
    plot = HeatmapPlot(
        data,
        config,
        position=X,
        color=Selector([U, P]),
        controls=[SliderAxis(batch)],
    )
    return Figure(plot, renderer=PlotlyRenderer(mathjax=mathjax))


class TestRenderer:
    def test_with_mathjax_titles_are_one_math_text(self):
        figure = _figure(mathjax=True)
        figure.draw(embedded=figure.controls)
        layout = figure.backend_figure.layout
        assert layout.xaxis.title.text == "$x_1$"
        (button, _) = layout.updatemenus[0].buttons
        assert button.label == "<i>u</i>"  # no math in buttons

    def test_without_mathjax_titles_are_plotly_markup(self):
        figure = _figure(mathjax=False)
        backend = figure.draw()
        assert backend.layout.xaxis.title.text == "<i>x</i><sub>1</sub>"

    def test_a_fixed_control_in_a_cell_title(self):
        config = DataConfiguration(
            GeometryAxes(GridGeometry.build_unit_grid(Variable("t", 1), 3)),
            GeometryAxes(GridGeometry.build_unit_grid(X, 4)),
            FeatureAxes(U),
        )
        data = np.zeros(config.shape)
        plot = HeatmapPlot(
            data, config, position=X, color=U,
            controls=[FixedAxis(Variable("t", 1), init_state=1)],
        )
        backend = Figure(plot, renderer=PlotlyRenderer()).draw()
        (title,) = [a.text for a in backend.layout.annotations]
        assert title.startswith("$") and title.endswith("$")

    def test_html_loads_mathjax_only_with_mathjax(self):
        with_mathjax = _figure(mathjax=True)
        with_mathjax.draw()
        without = _figure(mathjax=False)
        without.draw()
        assert "MathJax.js" in with_mathjax.renderer.html(full_html=False)
        assert "MathJax.js" not in without.renderer.html(full_html=False)

    def test_show_in_a_notebook_displays_html_with_mathjax(self, monkeypatch):
        from qewton.visualization.renderers import plotly as plotly_renderer

        shown = []
        monkeypatch.setattr(plotly_renderer, "_in_notebook", lambda: True)
        monkeypatch.setattr(
            "IPython.display.display", lambda obj: shown.append(obj.data)
        )
        figure = _figure(mathjax=True)
        figure.show()
        (page,) = shown
        assert "MathJax.js" in page


class TestDash:
    def test_labels_render_math_with_mathjax(self):
        pytest.importorskip("dash")
        from dash import dcc, html

        slider = SliderAxis(Variable("t", 1))
        assert isinstance(DashApplication.label(slider, mathjax=True), dcc.Markdown)
        label = DashApplication.label(slider, mathjax=False)
        assert isinstance(label, html.Label) and label.children == "t"


def test_image_export_without_kaleido_raises_a_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_kaleido(name, *args, **kwargs):
        if name == "kaleido":
            raise ImportError("no kaleido")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_kaleido)
    figure = _figure(mathjax=True)
    with pytest.raises(ImportError, match="pip install kaleido"):
        figure.save_png("out.png")
