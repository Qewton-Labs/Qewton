import qewton
import numpy as np
import qewton.visualization as viz

# dir
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

# Variables
X = qewton.Variable("x", 3)
N = qewton.Variable("n", 3)  # normal vectors
U = qewton.Variable("u", 3)  # velocity is vector valued
P = qewton.Variable("p", 1)  # pressure is scalar

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
outflow_sampler = qewton.RandomUniformSampler(
    outflow_geo, 8000, compute_normals=True, normal_name=N
)
side_sampler = qewton.RandomUniformSampler(side_geo, 15000)


def momentum_residual(u: U, p: P, x: X):  # type: ignore
    return mu * u.sym_grad(x).matrix_div(x) - p.gradient(x)


def mass_residual(u: U, x: X):  # type: ignore
    return u.div(x)


def inflow_residual(u: U, x: X):  # type: ignore
    inflow_function = qewton.bb.ZerosLike()(x)
    inflow_function[:, 1] = -((x[:, 0] - 1.0) ** 2 + x[:, 2] ** 2 - 0.01) / 0.01
    return u - inflow_function


def no_slip_residual(u: U):  # type: ignore
    return u


def outflow_residual(p: P):  # type: ignore
    return p


train = False
fake_training = True
save = False
MODEL_FILE_NAME = "stokes_model_trained"
if train:
    model = qewton.FCN(
        in_neurons=X,
        hidden_neurons=40,
        out_neurons=U + P,
        n_hidden_layers=6,
        activation=qewton.bb.Tanh,
    )
else:
    model = qewton.load(MODEL_FILE_NAME)

# Graphs
momentum_graph = qewton.PINNPipeline(
    volume_sampler,
    [model],
    residual=momentum_residual,
    residual_name="MomentumConstraint",
    weight=100.0,
)
momentum_constraint = momentum_graph.constraint

mass_graph = qewton.PINNPipeline(
    volume_sampler,
    [model],
    residual=mass_residual,
    residual_name="MassConstraint",
    weight=500.0,
)
mass_constraint = mass_graph.constraint

in_graph = qewton.PINNPipeline(
    inflow_sampler,
    [model],
    residual=inflow_residual,
    residual_name="InConstraint",
    weight=1.0,
)
in_constraint = in_graph.constraint

no_slip_graph = qewton.PINNPipeline(
    side_sampler,
    [model],
    residual=no_slip_residual,
    residual_name="SlipConstraint",
    weight=100.0,
)
no_slip_constraint = no_slip_graph.constraint

out_graph = qewton.PINNPipeline(
    outflow_sampler,
    [model],
    residual=outflow_residual,
    residual_name="OutConstraint",
    weight=1.0,
)
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
        # device=qewton.cuda(1),
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
        # device=qewton.cuda(1),
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
        phase_to_start_cache=lbfgs_phase,
    )
    trainer = qewton.optim.GraphBasedTrainer(
        optimization_phases=[adam_phase] if fake_training else [adam_phase, lbfgs_phase],
        graphs=[momentum_graph, mass_graph, in_graph, out_graph, no_slip_graph],
        # graphs=[momentum_graph, in_graph, out_graph, no_slip_graph],
        callbacks=[fix_sampler_callback],
        training_objectives=[
            momentum_constraint,
            mass_constraint,
            in_constraint,
            out_constraint,
            no_slip_constraint,
        ],
        # device=qewton.cuda(1),
    )

    trainer.run()
    if save:
        qewton.save(model, MODEL_FILE_NAME)


# Vis
if True:
    import torch
    import numpy as np

    fem_velocity = np.load("./data/stokes_pipe/velocity.npy")
    mesh_coords = np.load("./data/stokes_pipe/mesh_coords.npy").astype(np.float32)

    model.to("cpu")
    model_out = model(torch.tensor(mesh_coords))[:, :3].detach().cpu().numpy()

    fem_geometry = qewton.geometries.PointCloud(X, mesh_coords)
    velocity_config = qewton.DataConfiguration(
        qewton.GeometryAxes(fem_geometry), qewton.FeatureAxes(U)
    )

    reference_plot = qewton.visualization.auto_plot(
        fem_velocity, velocity_config, title="FEM Reference"
    )
    prediction_plot = qewton.visualization.auto_plot(
        model_out, velocity_config, title="PINN Prediction"
    )

    error_point_wise = np.linalg.norm(
        model_out - fem_velocity, ord=2, axis=1, keepdims=True
    ).astype(np.float32)
    ERROR = qewton.Variable("error", 1)
    error_config = qewton.DataConfiguration(
        qewton.GeometryAxes(fem_geometry), qewton.FeatureAxes(ERROR)
    )
    error_plot = qewton.visualization.auto_plot(
        error_point_wise, error_config, title="Point-wise Error"
    )

    fig = qewton.visualization.Figure(
        qewton.visualization.Row(reference_plot, prediction_plot, error_plot)
    )

    velocity_l2 = np.linalg.norm(fem_velocity, ord=2, axis=1)
    mse_error = np.mean(error_point_wise[:, 0] ** 2)
    print("MSE between PINN and FEM:", mse_error)
    print(
        "Relative MSE:", np.mean((error_point_wise[:, 0] / (velocity_l2 + 1.0e-2)) ** 2)
    )
    app = qewton.viz.DashApplication.create(fig)
    app.run(debug=True, jupyter_mode="external", use_reloader=False)
