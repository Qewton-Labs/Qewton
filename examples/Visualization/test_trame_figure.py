import qewton
import numpy as np
import sys
import qewton.visualization as viz

# Variables
X = qewton.Variable("x", 3)
U = qewton.Variable("velocity", 3)  # velocity is vector valued
P = qewton.Variable("pressure", 1)  # pressure is scalar

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
mesh_config = qewton.DataConfiguration(qewton.GeometryAxes(geo), qewton.FeatureAxes(X))
vertices = geo.mesh.vertices.numpy()
field = np.zeros((len(vertices), 1))
field = np.sin(vertices[:, 0]) * np.cos(vertices[:, 1]) + 10 * vertices[:, 2]
field = (np.sin(vertices[:, 0]) * np.cos(vertices[:, 1]) + 10 * vertices[:, 2])[:, None]
mesh_plot = viz.MeshFieldPlot(field, mesh_config, viz.ColorSpec(X.children[0]))

inflow_mesh_verts = qewton.geometries.PointCloud(X, inflow_geo.mesh.vertices)
inflow_mesh_config = qewton.DataConfiguration(
    qewton.GeometryAxes(inflow_mesh_verts), qewton.FeatureAxes(X)
)
inflow_geo_plot = viz.PointCloudPlot(inflow_geo.mesh.vertices, inflow_mesh_config)

# figure = viz.Figure(viz.Overlay(mesh_plot, inflow_geo_plot))
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

# Trame VTK figure 3D
if False:
    S = qewton.Variable("scalar_field", 1)
    data = np.concat((field, fem_velocity, fem_pressure), axis=1)
    scalar_fields = {
        "scalar_field": field,
        "pressure": fem_pressure,
        "s_velocity": np.linalg.norm(fem_velocity, axis=1)[:, None],
        "s_velocity_X": fem_velocity[:, 0:1],
        "s_velocity_Y": fem_velocity[:, 1:2],
        "s_velocity_Z": fem_velocity[:, 2:3],
    }
    vector_fields = {"velocity": fem_velocity}
    geometry = qewton.GeometryAxes(mesh_geo)
    fields = qewton.FeatureAxes(S * U * P)
    dc = qewton.DataConfiguration(geometry, fields)
    app = viz.TrameApplication.createMesh(dc, data)
    app.server.start()
    sys.exit(0)


# There is probably already something like qewton.geometries.GridGeometry to do this,
# but I couldn't find a MeshGeometry
def make_grid_mesh(coords, single_boundary=False):
    Nx, Ny, d = coords.shape
    lin_coord = lambda i, j: i * Ny + j
    flatten = lambda cells: [[lin_coord(i, j) for i, j in c] for c in cells]

    vertices = coords.reshape(Nx * Ny, d)
    cs  = [((i, j), (i+1, j), (i, j+1)) for i in range(Nx-1) for j in range(Ny-1)] # fmt: skip
    cs += [((i+1, j), (i+1, j+1), (i, j+1)) for i in range(Nx-1) for j in range(Ny-1)] # fmt: skip
    cells = flatten(cs)
    fs = [((i, 0), (i + 1, 0)) for i in range(Nx - 1)]
    fs += [((i, Ny - 1), (i + 1, Ny - 1)) for i in range(Nx - 1)]
    fs += [((0, i), (0, i + 1)) for i in range(Ny - 1)]
    fs += [((Nx - 1, i), (Nx - 1, i + 1)) for i in range(Ny - 1)]
    faces = flatten(fs)
    cell_markers = [1] * len(cells)
    if single_boundary:
        face_markers = [1] * (len(faces))
        marker_labels = {"boundary": (1, 1), "inside": (1, 2)}
    else:
        face_markers = [1] * (Nx - 1) + [2] * (Nx - 1) + [3] * (Ny - 1) + [4] * (Nx - 1)
        marker_labels = {
            "left": (1, 1),
            "right": (2, 1),
            "bottom": (3, 1),
            "top": (4, 1),
            "inside": (1, 2),
        }
    return qewton.geometries.Mesh(
        vertices, cells, cell_markers, faces, face_markers, marker_labels
    )


# Trame VTK figure 2D
if True:
    import torch

    angles = torch.load("examples/PINN/data/stokes_angle/angles.pt").numpy()
    coords = torch.load("examples/PINN/data/stokes_angle/coords.pt").numpy()
    velocities = torch.load("examples/PINN/data/stokes_angle/velocity.pt").numpy()

    flat_coords = coords[0, :, :, :]

    X2 = qewton.Variable("x", 2)
    # mesh_geo = qewton.geometries.GridGeometry(variable=X2, point_grid=flat_coords)
    mesh = make_grid_mesh(flat_coords)
    mesh_geo = qewton.geometries.MeshGeometry(variable=X2, mesh=mesh)

    A = qewton.Variable("angle", 3)
    V = qewton.Variable("velocity", 2)
    data = np.concat((angles.reshape(3, -1).T, velocities[0].reshape(-1, 2)), axis=1)
    geometry = qewton.GeometryAxes(mesh_geo)
    fields = qewton.FeatureAxes(A * V)
    dc = qewton.DataConfiguration(geometry, fields)
    app = viz.TrameApplication.createMesh(dc, data)
    app.server.start()
    sys.exit(0)
