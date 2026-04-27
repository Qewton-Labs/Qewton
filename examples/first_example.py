import torch
import pioneer

x_data = torch.linspace(0, 1, 1000).reshape(-1, 1)
u_data = x_data**2 + torch.sin(6.0 * x_data)
data = torch.column_stack((x_data, u_data))

X = pioneer.config.Variable("x", 1)
U = pioneer.config.Variable("u", 1)
dataset_X = pioneer.data.DataSet.from_data(x_data, X, batch_size=1000)
dataset_U = pioneer.data.DataSet.from_data(u_data, U, batch_size=1000)

model = pioneer.algorithms.FCN(
    in_neurons=1,
    hidden_neurons=50,
    out_neurons=1,
    n_hidden_layers=1,
    activation=pioneer.building_blocks.Tanh,
)

constraint = pioneer.constraints.MSEConstraint(
    model.output_ports[0].data_configuration,
)

computation_graph = pioneer.Graph()

computation_graph.connect(dataset_X, model)
computation_graph.connect(model, constraint.input_1)
computation_graph.connect(dataset_U, constraint.input_2)

computation_graph.setup()
print(computation_graph.sorted_nodes)


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
