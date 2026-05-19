import torch
import pioneer

x_data = torch.linspace(0, 1, 1000).reshape(-1, 1)
f_data = 2.0 * x_data
combined_data = torch.column_stack((x_data, f_data))

X = pioneer.config.Variable("x", 1)
U = pioneer.config.Variable("u", 1)
F = pioneer.config.Variable("f", 1)

x_config = pioneer.config.DataConfiguration(
    pioneer.config.BatchAxes(pioneer.config.AxesDim(None)),
    pioneer.config.FeatureAxes(X),
)
f_config = pioneer.config.DataConfiguration(
    pioneer.config.BatchAxes(pioneer.config.AxesDim(None)),
    pioneer.config.FeatureAxes(F),
)

dataset = pioneer.data.ArrayLikeDataSet(
    data=[x_data, f_data], data_configs=[x_config, f_config]
)

data_loader = pioneer.data.DataLoader(
    data_set=dataset,
    batch_size=1000,
    splitting_ratio=(1.0, 0.0, 0.0),
    shuffle_data=False,
)

model = pioneer.algorithms.FCN(
    in_neurons=X,
    hidden_neurons=50,
    out_neurons=U,
    n_hidden_layers=1,
    activation=pioneer.building_blocks.Tanh,
)


def residual_fun(u: U, f: F, x: X):  # type: ignore
    return u.gradient(x) - f


constraint = pioneer.constraints.PINNConstraint(residual_fun, name="PINNConstraint")
grad_tracking = pioneer.algorithms.building_blocks.GradientTracking()

# pipeline = pioneer.graphs.PINNPipeline(constraint / residual_fun, model, sampler)

computation_graph = pioneer.Graph()

with computation_graph.tracker():
    x, f = data_loader()
    x = grad_tracking(x)
    u = model(x)
    constraint(u, f, x)

# computation_graph.setup()

# Initial data:
initial_x_data = torch.zeros((1, 1))

initial_dataset = pioneer.data.ArrayLikeDataSet(
    data=[initial_x_data], data_configs=[x_config]
)
initial_data_loader = pioneer.data.DataLoader(data_set=initial_dataset, batch_size=1)


def initial_residual_fun(u: U):  # type: ignore
    return u


initial_constraint = pioneer.constraints.PINNConstraint(
    initial_residual_fun, name="InitialConstraint"
)

initial_graph = pioneer.Graph()
with initial_graph.tracker():
    x = initial_data_loader()
    u = model(x)
    initial_constraint(u)

###################################

adam_phase = pioneer.optim.OptimizationPhase(
    optimizer=pioneer.optim.Adam(),
    lr=0.001,
    max_iterations=2000,
)

lbfgs_phase = pioneer.optim.OptimizationPhase(
    optimizer=pioneer.optim.LBFGS(),
    lr=0.1,
    max_iterations=50,
    optimizer_args={"max_eval": 10},
)

trainer = pioneer.optim.GraphBasedTrainer(
    optimization_phases=[adam_phase, lbfgs_phase],
    graphs=[initial_graph, computation_graph],
    training_constraints=[initial_constraint, constraint],
    device="cuda:0",
)

trainer.run()

##########################################
model.to("cpu")
u = model(x_data)
import matplotlib.pyplot as plt

plt.plot(x_data.numpy(), u.detach().numpy(), label="Predicted")
plt.plot(x_data.numpy(), x_data.numpy() ** 2, label="True", linestyle="dashed")
plt.show()
plt.savefig("pinn_result.png")
