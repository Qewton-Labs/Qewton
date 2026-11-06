"""Registry of the visualization test matrix (test_matrix.py).

- CASES: data on one domain each.
- PLOTS: plot types with their bindings, and MODEL, the expected Result
  class per (plot, case); every other pair is an exclusion that must raise
  a ValueError at construction (EXCLUSIONS gives the reason).
- CONTROLS: what is applied on top: a control axis on an extra batch axis,
  a Selector, or a tuple of variables.

MODEL is written down independently of the plot classes' declarations, so
the test checks them instead of reading them.
"""

from dataclasses import dataclass

import numpy as np

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.geometries.discrete.point_cloud import PointCloud
from qewton.visualization import (
    BarPlot,
    FacetAxis,
    FixedAxis,
    HeatmapPlot,
    ImagePlot,
    LinePlot,
    OverlayAxis,
    ScatterPlot,
    Selector,
    SliderAxis,
    SurfacePlot,
    TimeAxis,
    VectorPlot,
)
from qewton.visualization.plots.result import (
    ArrowResult,
    BarResult,
    HeatmapResult,
    ImageResult,
    LineResult,
    MarkerResult,
    SurfaceResult,
)

T, X, Y = Variable("t", 1), Variable("x", 2), Variable("y", 3)
U, P, Q, R = Variable("u", 1), Variable("p", 1), Variable("q", 1), Variable("r", 2)
C = Variable("c", 3)  # RGB colors, or a vector in 3D
FEATURES = U * P * Q * R * C

#: Size of the extra axis a control acts on.
STEPS = 3


class NotApplicable(Exception):
    """The control has nothing to act on in this plot type."""


@dataclass
class Case:
    """Data on one domain (None: a batch axis) with features u, p, q, r, c."""

    name: str
    domain: object

    def build(self, extra_axis=None, seed=0, features=FEATURES, backend="numpy"):
        """Data and configuration, with an optional extra axis in front;
        the data as a numpy array or a torch tensor."""
        domain_axes = BatchAxes(6) if self.domain is None else GeometryAxes(self.domain)
        axes = ([extra_axis] if extra_axis is not None else []) + [
            domain_axes,
            FeatureAxes(features),
        ]
        config = DataConfiguration(*axes)
        data = np.random.default_rng(seed).random(config.shape).astype(np.float32) + 0.1
        if backend == "torch":
            import torch

            data = torch.as_tensor(data)
        return data, config, domain_axes


def _cases():
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    box = Box(Y, [0.0] * 3, 1.0, 1.0, 1.0).create_mesh(0.6)
    rng = np.random.default_rng(1)
    return [
        Case("points_1d", PointCloud(T, rng.random((9, 1)).astype(np.float32))),
        Case("points_2d", PointCloud(X, rng.random((12, 2)).astype(np.float32))),
        Case("points_3d", PointCloud(Y, rng.random((12, 3)).astype(np.float32))),
        Case("mesh_1d", Interval(T, 0.0, 1.0).create_mesh(0.25)),
        Case(
            "boundary_2d",
            MeshGeometry(X, square.create_mesh(0.5).mesh.get_boundary_mesh()),
        ),
        Case("mesh_2d", square.create_mesh(0.5)),
        Case("boundary_3d", MeshGeometry(Y, box.mesh.get_boundary_mesh())),
        Case("mesh_3d", box),
        Case("grid_1d", GridGeometry.build_unit_grid(T, 5)),
        Case("grid_2d", GridGeometry.build_unit_grid(X, 4)),
        Case("grid_3d", GridGeometry.build_unit_grid(Y, 3)),
        Case("batch", None),
    ]


CASES = {case.name: case for case in _cases()}

_ONE_D = ("points_1d", "mesh_1d", "grid_1d", "batch")
_MULTI_D = (
    "points_2d",
    "points_3d",
    "boundary_2d",
    "mesh_2d",
    "boundary_3d",
    "mesh_3d",
    "grid_2d",
    "grid_3d",
)

#: Expected Result class per (plot, case); every other pair is excluded.
MODEL = {
    ("heatmap", "grid_2d"): HeatmapResult,
    ("heatmap", "mesh_2d"): SurfaceResult,
    **{("line", c): LineResult for c in _ONE_D},
    **{("path", c): LineResult for c in _ONE_D + ("boundary_2d",)},
    **{("scatter_domain", c): MarkerResult for c in CASES},
    **{("scatter_values", c): MarkerResult for c in CASES},
    **{("bar", c): BarResult for c in _ONE_D},
    ("image", "grid_2d"): ImageResult,
    **{("surface_height", c): SurfaceResult for c in ("grid_2d", "mesh_2d")},
    **{
        ("surface_domain", c): SurfaceResult
        for c in ("boundary_3d", "mesh_3d", "grid_3d")
    },
    **{("vector", c): ArrowResult for c in _MULTI_D},
}

#: Why the pairs outside MODEL are excluded.
EXCLUSIONS = {
    "heatmap": "a heatmap draws a 2D domain flat",
    "line": "x is one coordinate of a one-dimensional domain",
    "path": "a path runs through a domain with an order (one-dimensional)",
    "bar": "x is one coordinate of a one-dimensional domain",
    "image": "an image is a rectilinear 2D grid",
    "surface_height": "a height lifts a 2D domain with cells",
    "surface_domain": "a surface in 3D needs cells of dimension 2 or 3 in 3D",
    "vector": "arrows need 2 or 3 position components",
}

PLOTS = [
    "heatmap",
    "line",
    "path",
    "scatter_domain",
    "scatter_values",
    "bar",
    "image",
    "surface_height",
    "surface_domain",
    "vector",
]
CONTROLS = ["none", "slider", "time", "facet", "fixed", "overlay", "selector", "tuple"]
_AXIS_CONTROLS = {
    "slider": SliderAxis,
    "time": TimeAxis,
    "facet": FacetAxis,
    "fixed": lambda axis: FixedAxis(axis, init_state=1),
    "overlay": OverlayAxis,
}


def expected(plot: str, case: str):
    """The expected Result class, or None for an exclusion."""
    return MODEL.get((plot, case))


def build(plot: str, case: Case, control: str, backend: str = "numpy"):
    """Build the plot of a matrix cell.

    Returns:
        tuple: The plot and the control (a ControlAxis or Selector), or None.

    Raises:
        NotApplicable: If the control has nothing to act on in the plot.
    """
    extra = BatchAxes(STEPS) if control in _AXIS_CONTROLS else None
    data, config, domain_axes = case.build(extra, backend=backend)
    axis_control = _AXIS_CONTROLS[control](extra) if extra is not None else None
    controls = [axis_control] if axis_control is not None else []
    selector = Selector([U, P]) if control == "selector" else None
    value = selector if selector is not None else U
    common = dict(controls=controls)

    if plot == "heatmap":
        _not_with(control, "tuple", "a heatmap has no y channel")
        return (
            HeatmapPlot(data, config, position=domain_axes, color=value, **common),
            axis_control or selector,
        )
    if plot == "line":
        y = (U, P) if control == "tuple" else value
        return (
            LinePlot(data, config, x=domain_axes, y=y, **common),
            axis_control or selector,
        )
    if plot == "path":
        _not_with(control, "tuple", "a path takes its positions from one feature")
        color = selector
        return (
            LinePlot(data, config, position=R, color=color, **common),
            axis_control or selector,
        )
    if plot == "scatter_domain":
        _not_with(control, "tuple", "positions from the domain take no tuple")
        return (
            ScatterPlot(data, config, position=domain_axes, color=value, **common),
            axis_control or selector,
        )
    if plot == "scatter_values":
        y = (P, Q) if control == "tuple" else P
        return (
            ScatterPlot(data, config, x=U, y=y, color=selector, **common),
            axis_control or selector,
        )
    if plot == "bar":
        y = (U, P) if control == "tuple" else value
        return (
            BarPlot(data, config, x=domain_axes, y=y, **common),
            axis_control or selector,
        )
    if plot == "image":
        _not_with(control, "tuple", "an image has no y channel")
        _not_with(control, "selector", "an image takes its colors from one feature")
        return (
            ImagePlot(data, config, position=domain_axes, color=C, **common),
            axis_control,
        )
    if plot == "surface_height":
        z = (U, P) if control == "tuple" else value
        return (
            SurfacePlot(data, config, position=domain_axes, z=z, color=selector, **common),
            axis_control or selector,
        )
    if plot == "surface_domain":
        _not_with(control, "tuple", "positions from the domain take no tuple")
        return (
            SurfacePlot(data, config, position=domain_axes, color=value, **common),
            axis_control or selector,
        )
    if plot == "vector":
        _not_with(control, "tuple", "positions from the domain take no tuple")
        vector = C if _n_positions(domain_axes) == 3 else R
        return (
            VectorPlot(
                data, config, position=domain_axes, vector=vector, color=selector, **common
            ),
            axis_control or selector,
        )
    raise KeyError(plot)


def _n_positions(domain_axes) -> int:
    return domain_axes.geometry.dim if isinstance(domain_axes, GeometryAxes) else 1


def _not_with(control: str, name: str, reason: str) -> None:
    if control == name:
        raise NotApplicable(reason)
