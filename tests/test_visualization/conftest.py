"""Shared fixtures for qewton.visualization tests.

A local conftest.py (unlike the rest of the test suite, which mostly
defines helpers inline per file) because visualization tests repeatedly
need the same handful of non-trivial objects - a real mesh, a real
computation graph, common Variables - across many otherwise-independent
test files.
"""

from typing import Annotated

import numpy as np
import pytest

from qewton.algorithms.building_blocks.creation import Zeros
from qewton.algorithms.dl_models.fcn import FCN
from qewton.backends import TensorType
from qewton.config.axes import EllipsisAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_2d.circle import Circle
from qewton.geometries.continuous.domains_3d.cylinder import Cylinder
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node


@pytest.fixture
def small_mesh_geometry():
    """A tiny, fast, hand-built 2D mesh (4 vertices, 2 triangles) - a unit
    square split along one diagonal. Cheap enough to use in tests that don't
    care about mesh realism, only structure."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cells = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(vertices=vertices, cells=cells)
    return MeshGeometry(Variable("p", 2), mesh)


@pytest.fixture
def circle_mesh_geometry():
    """A real discretized 2D geometry (Circle), for tests that need
    something less trivial than small_mesh_geometry - e.g. enough vertices
    for a field to look like a field."""
    x2 = Variable("x", 2)
    circle = Circle(variable=x2, center=[0, 0], radius=1.0)
    return circle.create_mesh(max_vertex_distance=0.3)


@pytest.fixture
def cylinder_mesh_geometry():
    """A real discretized 3D volumetric geometry (Cylinder) - used by every
    test needing a genuine 3D mesh (surface extraction, vector fields,
    volume resampling)."""
    x3 = Variable("x", 3)
    cylinder = Cylinder(variable=x3, center=[0, 0, 0], radius=1.0, height=2.0)
    return cylinder.create_mesh(max_vertex_distance=0.3)


class _LossNode(Node):
    """Minimal leaf Node (pass-through) for building a tiny real Graph in
    tests - Annotated[TensorType, DataConfiguration(EllipsisAxes())] mirrors
    the pattern qewton.algorithms.building_blocks.creation.Zeros itself
    uses, so it wires up against a Zeros source / FCN without a data-type
    mismatch (see graphs.py's connect() config unification)."""

    def forward(
        self, pred: Annotated[TensorType, DataConfiguration(EllipsisAxes())]
    ) -> Annotated[TensorType, DataConfiguration(EllipsisAxes())]:
        return pred


@pytest.fixture
def simple_graph():
    """A minimal real, sorted Graph: Source (Zeros) -> FCN -> Loss.
    Small enough to lay out fast, but exercises a genuine composite
    (GraphNode) for GraphPlot/GraphLayout tests."""
    graph = Graph()
    source = Zeros(shape=(1,), name="Source")
    fcn = FCN(
        in_neurons=Variable("x", 1),
        hidden_neurons=5,
        out_neurons=Variable("u", 1),
        n_hidden_layers=2,
        name="fcn",
    )
    loss = _LossNode(name="Loss")
    graph.connect(source.output_ports[0], fcn.input_ports[0])
    graph.connect(fcn.output_ports[0], loss.input_ports[0])
    graph.sort()
    return graph
