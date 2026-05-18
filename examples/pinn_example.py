import torch
import pioneer

x_data = torch.linspace(0, 1, 1000).reshape(-1, 1)
f_data = 0.5 * x_data
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


constraint = pioneer.constraints.PINNConstraint(residual_fun)
grad_tracking = pioneer.algorithms.building_blocks.GradientTracking()

# pipeline = pioneer.graphs.PINNPipeline(constraint / residual_fun, model, sampler)

computation_graph = pioneer.Graph()

with computation_graph.tracker():
    x, f = data_loader()
    x = grad_tracking(x)
    u = model(x)
    constraint(u, f, x)

# computation_graph.setup()

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
    graphs=[computation_graph],
    training_constraints=[constraint],
    device="cuda:0",
)

trainer.run()
