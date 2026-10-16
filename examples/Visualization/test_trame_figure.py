import qewton
import numpy as np
import sys
import qewton.visualization as viz

# Trame
from trame.app import TrameApp
from trame.decorators import change
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3, plotly

import plotly.express as px
import plotly.graph_objects as go

# Variables
X = qewton.Variable("x", 3)
U = qewton.Variable("u", 3) # velocity is vector valued
P = qewton.Variable("p", 1) # pressure is scalar

# FEM reference
fem_velocity = np.load("examples/PINN/data/stokes_pipe/velocity.npy")
fem_pressure = np.load("examples/PINN/data/stokes_pipe/pressure.npy")


# Geometry
mesh_geo = qewton.geometries.MeshGeometry.load_mesh(
    variable=X, file_path="examples/PINN/data/stokes_pipe/u_pipe.msh"
)

inflow_geo = mesh_geo.boundary.get_submesh("Inflow")
outflow_geo = mesh_geo.boundary.get_submesh("Outflow")
side_geo = mesh_geo.boundary.get_submesh("Side")

# Figure
geo = mesh_geo
mesh_config = qewton.DataConfiguration(
    qewton.GeometryAxes(geo), qewton.FeatureAxes(X)
)
vertices = geo.mesh.vertices.numpy()
field = np.zeros((len(vertices), 1))
field = (np.sin(vertices[:, 0]) * np.cos(vertices[:, 1]) + 10*vertices[:, 2])
field = (np.sin(vertices[:, 0]) * np.cos(vertices[:, 1]) + 10*vertices[:, 2])[:, None]
mesh_plot = viz.MeshFieldPlot(field, mesh_config, viz.ColorSpec(X.children[0]))

inflow_mesh_verts = qewton.geometries.PointCloud(X, inflow_geo.mesh.vertices)
inflow_mesh_config = qewton.DataConfiguration(
    qewton.GeometryAxes(inflow_mesh_verts), qewton.FeatureAxes(X)
)
inflow_geo_plot = viz.PointCloudPlot(inflow_geo.mesh.vertices, inflow_mesh_config)

#figure = viz.Figure(viz.Overlay(mesh_plot, inflow_geo_plot))
figure = viz.Figure(viz.Row(mesh_plot, inflow_geo_plot))

# Dash plotly figure
if False:
    app = viz.DashApplication.create(figure)
    app.run(debug=True, use_reloader=False)
    sys.exit(0)

# Trame plotly figure
if False:
    app = viz.TrameApplication.create(figure)
    app.server.start()
    sys.exit(0)

# Trame VTK figure
if True:
    scalar_fields = {"scalar_field": field,
                     "pressure": fem_pressure,
                     "Xvelocity": np.linalg.norm(fem_velocity, axis=1)[:, None],
                     "Xvelocity_X": fem_velocity[:, 0:1],
                     "Xvelocity_Y": fem_velocity[:, 1:2],
                     "Xvelocity_Z": fem_velocity[:, 2:3]}
    vector_fields = {"velocity": fem_velocity}
    app = viz.TrameApplication.createMesh(mesh_geo, scalar_fields, vector_fields)
    app.server.start()
    sys.exit(0)