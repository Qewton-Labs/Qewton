import numpy as np
import pytest

from qewton.config.variables import Variable
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.visualization.figure import Figure
from qewton.visualization.plots.geometry import GeometryPlot


class TestGeometryPlot:
    def test_2d_geometry_uses_geometryartist2d(self, small_mesh_geometry):
        plot = GeometryPlot(small_mesh_geometry)
        assert plot.dim == 2
        assert plot.embedding_dim == 2
        backend_figure = Figure(plot).draw()
        assert backend_figure.data[0].type == "scatter"  # filled triangulation

    def test_3d_geometry_uses_geometryartist(self, cylinder_mesh_geometry):
        plot = GeometryPlot(cylinder_mesh_geometry)
        assert plot.dim == 3
        assert plot.embedding_dim == 3
        backend_figure = Figure(plot).draw()
        assert backend_figure.data[0].type == "mesh3d"

    def test_stores_the_geometry_for_axis_naming(self, small_mesh_geometry):
        """Regression: GeometryPlot used to never store the Geometry it was
        given at all, so its artist had no way to reach axis names."""
        plot = GeometryPlot(small_mesh_geometry)
        assert plot.geometry is small_mesh_geometry

    def test_2d_axis_titles_from_the_geometry_variable(self, small_mesh_geometry):
        plot = GeometryPlot(small_mesh_geometry)
        backend_figure = Figure(plot).draw()
        assert backend_figure.layout.xaxis.title.text == "$p_1$"
        assert backend_figure.layout.yaxis.title.text == "$p_2$"

    def test_3d_axis_titles_from_the_geometry_variable(self, cylinder_mesh_geometry):
        plot = GeometryPlot(cylinder_mesh_geometry)
        backend_figure = Figure(plot).draw()
        titles = {
            backend_figure.layout.scene.xaxis.title.text,
            backend_figure.layout.scene.yaxis.title.text,
            backend_figure.layout.scene.zaxis.title.text,
        }
        assert titles == {"$x_1$", "$x_2$", "$x_3$"}

    def test_volumetric_mesh_splits_boundary_from_interior(self, cylinder_mesh_geometry):
        plot = GeometryPlot(cylinder_mesh_geometry)
        assert plot.interior_mesh is not None
        assert plot.boundary_mesh is not None

    def test_unsupported_dimension_raises(self):
        vertices = np.array([[0.0], [1.0]])
        cells = np.array([[0, 1]])
        mesh = Mesh(vertices=vertices, cells=cells)
        mgeo = MeshGeometry(Variable("p", 1), mesh)
        with pytest.raises(NotImplementedError):
            GeometryPlot(mgeo)

    def test_redraw_does_not_duplicate_traces(self, small_mesh_geometry):
        fig = Figure(GeometryPlot(small_mesh_geometry))
        backend_figure = fig.draw()
        n_traces = len(backend_figure.data)
        fig.draw()
        assert len(backend_figure.data) == n_traces
