import torch

# model = torch.nn.Sequential(torch.nn.Linear(10, 10))
# model_b = torch.nn.Linear(3, 3)


import qewton

graph = qewton.graphs.Graph()

x_data = torch.linspace(0, 1, 1000).reshape(-1, 1)
u_data = x_data**2 + torch.sin(6.0 * x_data)
data = torch.column_stack((x_data, u_data))

X = qewton.config.Variable("x", 1)
U = qewton.config.Variable("u", 1)
dataset_X = qewton.data.DataSet.from_data(x_data, X, batch_size=1000)
dataset_U = qewton.data.DataSet.from_data(u_data, U, batch_size=1000)

model = qewton.algorithms.FCN(1, 50, 1, 1, activation=qewton.building_blocks.Tanh)
mse_constraint = qewton.constraints.MSEConstraint(
    model.output_ports[0].data_configuration
)

with graph.tracker():
    x = dataset_X()
    u = dataset_U()
    model_out = model(x)
    final_out = mse_constraint(model_out, u)

adam_phase = qewton.optim.OptimizationPhase(
    optimizer=qewton.optim.Adam(),
    lr=0.001,
    max_iterations=2000,
)

lbfgs_phase = qewton.optim.OptimizationPhase(
    optimizer=qewton.optim.LBFGS(),
    lr=0.1,
    max_iterations=50,
    optimizer_args={"max_eval": 10},
)

trainer = qewton.optim.GraphBasedTrainer(
    optimization_phases=[adam_phase, lbfgs_phase],
    graphs=[graph],
    training_constraints=[mse_constraint],
    device="cuda:0",
)

trainer.run()
