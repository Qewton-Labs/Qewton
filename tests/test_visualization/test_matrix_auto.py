"""Test matrix of auto_plot: domain × value × leftover axis.

The domains are the cases of the plot matrix (matrix_cases.py). AUTO
gives the expected plot type per (domain kind, value); every other pair
must raise a ValueError, for the reason in EXCLUSIONS. A leftover axis in
front of the data gets the default control, except for values drawn
against each other without a domain, where every sample is one element.
Products and several geometries follow as a table of named setups.
"""

import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization import (
    Figure,
    HeatmapPlot,
    LinePlot,
    ScatterPlot,
    Selector,
    SliderAxis,
    SurfacePlot,
    TimeAxis,
    VectorPlot,
    auto_plot,
)

from matrix_cases import CASES, C, P, Q, R, T, U, X, Y
from recording_renderer import RecordingRenderer

#: The kind of each domain of the plot matrix, as auto_plot sees it.
KIND = {
    "points_1d": "1d",
    "mesh_1d": "1d",
    "grid_1d": "1d",
    "boundary_2d": "curve_2d",
    "points_2d": "points_2d",
    "mesh_2d": "area_2d",
    "grid_2d": "area_2d",
    "points_3d": "points_3d",
    "boundary_3d": "surface_3d",
    "mesh_3d": "volume_3d",
    "grid_3d": "volume_3d",
    "batch": "batch",
}

#: The values: the features of the data ("coordinates": the domain's own
#: variable).
VALUES = {
    "scalar": U,
    "scalars": U * P,
    "three_scalars": U * P * Q,
    "vector_2": R,
    "vector_3": C,
    "mixed": U * R,
    "coordinates": None,
}

#: Expected plot type per (domain kind, value); everything else raises.
AUTO = {
    **{("1d", v): LinePlot for v in ("scalar", "scalars", "three_scalars")},
    ("1d", "coordinates"): ScatterPlot,
    **{("curve_2d", v): ScatterPlot for v in ("scalar", "scalars", "three_scalars")},
    ("curve_2d", "vector_2"): VectorPlot,
    ("curve_2d", "coordinates"): ScatterPlot,
    **{("points_2d", v): ScatterPlot for v in ("scalar", "scalars", "three_scalars")},
    ("points_2d", "vector_2"): VectorPlot,
    ("points_2d", "coordinates"): ScatterPlot,
    **{("area_2d", v): HeatmapPlot for v in ("scalar", "scalars", "three_scalars")},
    ("area_2d", "vector_2"): VectorPlot,
    ("area_2d", "coordinates"): ScatterPlot,
    **{("points_3d", v): ScatterPlot for v in ("scalar", "scalars", "three_scalars")},
    ("points_3d", "vector_3"): VectorPlot,
    ("points_3d", "coordinates"): ScatterPlot,
    **{("surface_3d", v): SurfacePlot for v in ("scalar", "scalars", "three_scalars")},
    ("surface_3d", "vector_3"): VectorPlot,
    ("surface_3d", "coordinates"): ScatterPlot,
    **{("volume_3d", v): SurfacePlot for v in ("scalar", "scalars", "three_scalars")},
    ("volume_3d", "vector_3"): VectorPlot,
    ("volume_3d", "coordinates"): ScatterPlot,
    ("batch", "scalar"): LinePlot,
    ("batch", "three_scalars"): LinePlot,
    ("batch", "scalars"): ScatterPlot,  # two values against each other
    ("batch", "vector_2"): ScatterPlot,  # positions from the values
    ("batch", "vector_3"): ScatterPlot,
}

#: Why the other pairs raise.
EXCLUSIONS = {
    "mixed": "scalars and vectors cannot share one channel",
    "vector_2": "a vector needs one component per coordinate of the domain",
    "vector_3": "a vector needs one component per coordinate of the domain",
}

#: Values drawn against each other, without a domain: no controls.
WITHOUT_DOMAIN = {("batch", "scalars"), ("batch", "vector_2"), ("batch", "vector_3")}

LEFTOVERS = {"none": None, "slider": SliderAxis, "time": TimeAxis}


def _features(case, value):
    if value != "coordinates":
        return VALUES[value]
    if case.domain is None:
        pytest.skip("a batch has no coordinates")
    return case.domain.variable


@pytest.mark.parametrize("leftover", list(LEFTOVERS))
@pytest.mark.parametrize("value", list(VALUES))
@pytest.mark.parametrize("case_name", list(CASES))
def test_auto_plot(case_name, value, leftover):
    case = CASES[case_name]
    extra = BatchAxes(3) if LEFTOVERS[leftover] is not None else None
    data, config, _ = case.build(extra, features=_features(case, value))
    kwargs = dict(default_control=LEFTOVERS[leftover]) if extra is not None else {}
    expected = AUTO.get((KIND[case_name], value))
    if expected is None:
        assert value in EXCLUSIONS, f"no reason for excluding {value} on {case_name}"
        with pytest.raises(ValueError):
            auto_plot(data, config, **kwargs)
        return

    plot = auto_plot(data, config, **kwargs)
    assert type(plot) is expected
    without_domain = (KIND[case_name], value) in WITHOUT_DOMAIN
    if extra is None or without_domain:
        assert plot.controls == []
    else:
        assert [type(c) for c in plot.controls] == [LEFTOVERS[leftover]]
    quantities = config.feature_axes.variables.quantities
    if not without_domain and len(quantities) > 1:
        (selector,) = plot.selectors
        assert selector.candidates == quantities
    Figure(plot, renderer=RecordingRenderer()).draw()


# --- products and several geometries ----------------------------------------

M = Variable("m", 1)


def _on_product(geometry, features=U):
    mesh = geometry.create_mesh(0.5)
    config = DataConfiguration(GeometryAxes(mesh), FeatureAxes(features))
    return np.random.default_rng(0).random(config.shape), config


def _time():
    return Interval(T, 0.0, 2.0)


def _square():
    return Rectangle(X, [0.0, 0.0], 1.0, 1.0)


def _two_geometries(second):
    config = DataConfiguration(
        GeometryAxes(GridGeometry.build_unit_grid(T, 4)), GeometryAxes(second), FeatureAxes(U)
    )
    return np.random.default_rng(0).random(config.shape), config


def _wide_geometries():
    Z = Variable("z", 2)
    config = DataConfiguration(
        GeometryAxes(_square().create_mesh(0.5)),
        GeometryAxes(Rectangle(Z, [0, 0], 1, 1).create_mesh(0.5)),
        FeatureAxes(U),
    )
    return np.random.default_rng(0).random(config.shape), config


#: name: (data and configuration, controls, expected type, drawn variable,
#: number of controls); a type None raises.
SETUPS = {
    "product fits whole": (lambda: _on_product(_time() * _square()), [], SurfacePlot, None, 0),
    "control on a factor": (
        lambda: _on_product(_time() * _square()), [TimeAxis(T)], HeatmapPlot, X, 1
    ),
    "product beyond 3d": (
        lambda: _on_product(_time() * Box(Y, [0.0] * 3, 1, 1, 1)), [], SurfacePlot, Y, 1
    ),
    "vector splits off time": (
        lambda: _on_product(_square() * _time(), R), [], VectorPlot, X, 1
    ),
    "ambiguous factors": (
        lambda: _on_product(_time() * Interval(M, 1.0, 3.0) * _square()), [], None, None, 0
    ),
    "1d geometries get controls": (
        lambda: _two_geometries(_square().create_mesh(0.5)), [], HeatmapPlot, X, 1
    ),
    "two wide geometries": (_wide_geometries, [], None, None, 0),
}


@pytest.mark.parametrize("name", list(SETUPS))
def test_auto_plot_on_products_and_several_geometries(name):
    build, controls, expected, drawn, n_controls = SETUPS[name]
    data, config = build()
    if expected is None:
        with pytest.raises(ValueError):
            auto_plot(data, config, controls=controls)
        return
    plot = auto_plot(data, config, controls=controls)
    assert type(plot) is expected
    if drawn is not None:
        assert plot.domain.variable == drawn
    assert len(plot.controls) == n_controls
    Figure(plot, renderer=RecordingRenderer()).draw()
