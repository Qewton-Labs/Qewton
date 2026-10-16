import qewton
import numpy as np
import sys
import qewton.visualization as viz

# dir
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

# Variables
X = qewton.Variable("x", 3)
N = qewton.Variable("n", 3) # normal vectors
U = qewton.Variable("u", 3) # velocity is vector valued
P = qewton.Variable("p", 1) # pressure is scalar

mu = 0.001

# Geometry
mesh_geo = qewton.geometries.MeshGeometry.load_mesh(
    variable=X, file_path="data/stokes_pipe/u_pipe.msh"
)

inflow_geo = mesh_geo.boundary.get_submesh("Inflow")
outflow_geo = mesh_geo.boundary.get_submesh("Outflow")
side_geo = mesh_geo.boundary.get_submesh("Side")

# Samplers
volume_sampler = qewton.RandomUniformSampler(mesh_geo, 60000)
inflow_sampler = qewton.RandomUniformSampler(inflow_geo, 8000)
outflow_sampler = qewton.RandomUniformSampler(outflow_geo, 8000, compute_normals=True, normal_name=N)
side_sampler = qewton.RandomUniformSampler(side_geo, 15000)

# Use Dash to sample mesh vertices instead of samplers
if False:
    inflow_mesh_verts = qewton.geometries.PointCloud(X, inflow_geo.mesh.vertices)
    inflow_mesh_config = qewton.DataConfiguration(
        qewton.GeometryAxes(inflow_mesh_verts), qewton.FeatureAxes(X)
    )
    inflow_geo_plot = viz.PointCloudPlot(inflow_geo.mesh.vertices, inflow_mesh_config)

    outflow_mesh_verts = qewton.geometries.PointCloud(X, outflow_geo.mesh.vertices)
    outflow_mesh_config = qewton.DataConfiguration(
        qewton.GeometryAxes(outflow_mesh_verts), qewton.FeatureAxes(X)
    )
    outflow_geo_plot = viz.PointCloudPlot(outflow_geo.mesh.vertices, outflow_mesh_config)

    side_mesh_verts = qewton.geometries.PointCloud(X, side_geo.mesh.vertices)
    side_mesh_config = qewton.DataConfiguration(
        qewton.GeometryAxes(side_mesh_verts), qewton.FeatureAxes(X)
    )
    side_geo_plot = viz.PointCloudPlot(side_geo.mesh.vertices, side_mesh_config)

    mesh_verts = qewton.geometries.PointCloud(X, mesh_geo.mesh.vertices)
    mesh_config = qewton.DataConfiguration(
        qewton.GeometryAxes(mesh_verts), qewton.FeatureAxes(X)
    )
    mesh_geo_plot = viz.PointCloudPlot(mesh_geo.mesh.vertices, mesh_config)

    #geo = inflow_geo
    geo = mesh_geo
    mesh_config = qewton.DataConfiguration(
        qewton.GeometryAxes(geo), qewton.FeatureAxes(X)
    )
    #test = viz.auto_plot(geo, mesh_config)
    field = np.zeros((len(geo.mesh.vertices), 1))
    mesh_plot = viz.MeshFieldPlot(field, mesh_config, viz.ColorSpec(X.children[0]))


    #x2 = qewton.Variable("x", 2)
    #circle = qewton.geometries.continuous.domains_2d.circle.Circle(variable=x2, center=[0, 0], radius=1.0)
    #circle_mesh_geometry = circle.create_mesh(max_vertex_distance=0.3)

    #cylinder = qewton.geometries.continuous.domains_3d.cylinder.Cylinder(variable=X, center=[0, 0, 0], radius=1.0, height=2.0)
    #cylinder_mesh_geometry = cylinder.create_mesh(max_vertex_distance=0.3)

    #U = qewton.Variable("u", 1)
    ##geom = circle_mesh_geometry
    #geom = cylinder_mesh_geometry
    #vertices = np.asarray(geom.mesh.vertices)
    #field = (np.sin(vertices[:, 0]) * np.cos(vertices[:, 1]))[:, None]
    #config = qewton.DataConfiguration(qewton.GeometryAxes(geom), qewton.FeatureAxes(U))
    #plot = viz.MeshFieldPlot(field, config, color=viz.ColorSpec(U))
    #result = plot.evaluate()
    #fig = viz.Figure(plot)

    fig = qewton.visualization.Figure(
        #inflow_geo_plot
        #mesh_geo_plot
        mesh_plot
        #qewton.visualization.Overlay(mesh_geo_plot, inflow_geo_plot, outflow_geo_plot, side_geo_plot)
        #qewton.visualization.Overlay(inflow_geo_plot, outflow_geo_plot, side_geo_plot)
        #qewton.visualization.Overlay(inflow_geo_plot)
    )
    app = qewton.visualization.DashApplication.create(fig)
    app.run(debug=True, jupyter_mode="external", use_reloader=False)
    sys.exit(0)

# Point sampling vis with dash / plotly
if False:
    #import plotly.io as pio
    #pio.renderers.default = "vscode"

    inflow_plot = inflow_sampler.visualize().plots[0]
    outflow_plot = outflow_sampler.visualize().plots[0]
    #side_plot = side_sampler.visualize().plots[0]

    fig = qewton.visualization.Figure(
        qewton.visualization.Overlay(inflow_plot, outflow_plot)#, side_plot)
    )
    app = qewton.visualization.DashApplication.create(fig)
    app.run(debug=True, jupyter_mode="external", use_reloader=False)
    sys.exit(0)

def momentum_residual(u: U, p: P, x: X):  # type: ignore
    return mu * u.sym_grad(x).matrix_div(x) - p.gradient(x)

def mass_residual(u: U, x: X):  # type: ignore
    return u.div(x)

def inflow_residual(u: U, x: X):  # type: ignore
    inflow_function = qewton.bb.ZerosLike()(x)
    inflow_function[:, 1] = - ((x[:, 0]-1.0)**2 + x[:, 2]**2 - 0.01)/0.01
    return u - inflow_function

def no_slip_residual(u: U):  # type: ignore
    return u

def outflow_residual(p: P):  # type: ignore
    return p

train = False
fake_training = True
save = False
if train:
    model = qewton.FCN(
        in_neurons=X,
        hidden_neurons=40,
        out_neurons=U+P,
        n_hidden_layers=6,
        activation=qewton.bb.Tanh,
    )
else:
    model = qewton.load("stokes_model_unfinished")

# Graphs
momentum_graph = qewton.PINNPipeline(volume_sampler, [model], 
                                    residual=momentum_residual, 
                                    residual_name="MomentumConstraint", 
                                    weight=100.0)
momentum_constraint = momentum_graph.constraint

mass_graph = qewton.PINNPipeline(volume_sampler, [model], 
                                residual=mass_residual, 
                                residual_name="MassConstraint", 
                                weight=500.0)
mass_constraint = mass_graph.constraint

in_graph = qewton.PINNPipeline(inflow_sampler, [model], 
                                residual=inflow_residual, 
                                residual_name="InConstraint", 
                                weight=1.0)
in_constraint = in_graph.constraint

no_slip_graph = qewton.PINNPipeline(side_sampler, [model], 
                                    residual=no_slip_residual, 
                                    residual_name="SlipConstraint", 
                                    weight=100.0)
no_slip_constraint = no_slip_graph.constraint

out_graph = qewton.PINNPipeline(outflow_sampler, [model], 
                                residual=outflow_residual, 
                                residual_name="OutConstraint", 
                                weight=1.0)
out_constraint = out_graph.constraint

if train:
    # Train inflow boundary condition first:
    adam_phase = qewton.optim.OptimizationPhase(
        optimizer=qewton.optim.Adam(),
        lr=0.001,
        max_iterations=0 if fake_training else 100,
    )

    trainer = qewton.optim.GraphBasedTrainer(
        optimization_phases=[adam_phase],
        graphs=[in_graph],
        training_objectives=[in_constraint],
        #device=qewton.cuda(1),
    )
    trainer.run()

    # Train all other boundary condition:
    adam_phase = qewton.optim.OptimizationPhase(
        optimizer=qewton.optim.Adam(),
        lr=0.0005,
        max_iterations=0 if fake_training else 1000,
    )

    trainer = qewton.optim.GraphBasedTrainer(
        optimization_phases=[adam_phase],
        graphs=[in_graph, out_graph, no_slip_graph],
        training_objectives=[in_constraint, out_constraint, no_slip_constraint],
        #device=qewton.cuda(1),
    )

    trainer.run()

    # More
    adam_phase = qewton.optim.OptimizationPhase(
        optimizer=qewton.optim.Adam(),
        lr=0.0001,
        max_iterations=0 if fake_training else 5000,
    )
    lbfgs_phase = qewton.optim.OptimizationPhase(
        optimizer=qewton.optim.LBFGS(),
        lr=0.1,
        max_iterations=0 if fake_training else 1000,
        optimizer_args={"max_eval": 1 if fake_training else 20},
    )
    fix_sampler_callback = qewton.optim.CacheDataCallback(
        data_nodes=[volume_sampler, inflow_sampler, side_sampler, outflow_sampler],
        phase_to_start_cache=lbfgs_phase
    )
    trainer = qewton.optim.GraphBasedTrainer(
        optimization_phases=[adam_phase] if fake_training else [adam_phase, lbfgs_phase],
        graphs=[momentum_graph, mass_graph, in_graph, out_graph, no_slip_graph],
        #graphs=[momentum_graph, in_graph, out_graph, no_slip_graph],
        callbacks=[fix_sampler_callback],
        training_objectives=[momentum_constraint, mass_constraint, in_constraint, 
                            out_constraint, no_slip_constraint],
        #device=qewton.cuda(1),
    )

    trainer.run()
    if save:
        qewton.save(model, "stokes_model_trained")



# new visualization way:
if False:
    fem_velocity = np.load("./data/stokes_pipe/velocity.npy")
    mesh_coords = np.load("./data/stokes_pipe/mesh_coords.npy").astype(np.float32)

    fem_geometry = qewton.geometries.PointCloud(X, mesh_coords)
    velocity_config = qewton.DataConfiguration(qewton.GeometryAxes(fem_geometry), qewton.FeatureAxes(U))


    fig = qewton.viz.Figure(
        momentum_graph.visualize(model.output_ports[0],
                                variables=[U],
                                reference=fem_velocity,
                                reference_config=velocity_config,
                                )
        )
    # could be improved by using colors etc, but for now its working

    app = qewton.viz.DashApplication.create(fig)
    app.run(debug=True, jupyter_mode="external", use_reloader=False)

if True:
    import torch
    import numpy as np

    fem_velocity = np.load("./data/stokes_pipe/velocity.npy")
    mesh_coords = np.load("./data/stokes_pipe/mesh_coords.npy").astype(np.float32)

    model.to("cpu")
    model_out = model(torch.tensor(mesh_coords))[:, :3].detach().cpu().numpy()

    fem_geometry = qewton.geometries.PointCloud(X, mesh_coords)
    velocity_config = qewton.DataConfiguration(qewton.GeometryAxes(fem_geometry), qewton.FeatureAxes(U))

    reference_plot = qewton.visualization.auto_plot(fem_velocity, velocity_config, title="FEM Reference")
    prediction_plot = qewton.visualization.auto_plot(model_out, velocity_config, title="PINN Prediction")

    error_point_wise = np.linalg.norm(model_out - fem_velocity, ord=2, axis=1, keepdims=True).astype(np.float32)
    ERROR = qewton.Variable("error", 1)
    error_config = qewton.DataConfiguration(qewton.GeometryAxes(fem_geometry), qewton.FeatureAxes(ERROR))
    error_plot = qewton.visualization.auto_plot(error_point_wise, error_config, title="Point-wise Error")

    fig = qewton.visualization.Figure(
        qewton.visualization.Row(reference_plot, prediction_plot, error_plot)
    )
    app = qewton.viz.DashApplication.create(fig)
    app.run(debug=True, jupyter_mode="external", use_reloader=False)

    velocity_l2 = np.linalg.norm(fem_velocity, ord=2, axis=1)
    mse_error = np.mean(error_point_wise[:, 0] ** 2)
    print("MSE between PINN and FEM:", mse_error)
    print("Relative MSE:", np.mean((error_point_wise[:, 0] / (velocity_l2 + 1.e-2)) ** 2))

if False:
    import torch
    inflow_sampler = qewton.RandomUniformSampler(inflow_geo, 1000)
    p = inflow_sampler()
    u = model(p)[:, :3]
    u_in = torch.zeros_like(p)
    u_in[:, 1] = - ((p[:, 0]-1.0)**2 + p[:, 2]**2 - 0.01)/0.01

    speed = torch.linalg.norm(u, axis=1).detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
    SPEED = qewton.Variable("speed", 1)
    inflow_projection = qewton.geometries.PointCloud(
        qewton.Variable("x_proj", 2), p[:, [0, 2]].detach().cpu().numpy()
    )
    speed_config = qewton.DataConfiguration(
        qewton.GeometryAxes(inflow_projection), qewton.FeatureAxes(SPEED)
    )
    qewton.visualization.Figure(
        qewton.visualization.auto_plot(speed, speed_config, title="Inflow speed")
    ).show()