import torch
import pioneer

x_data = torch.linspace(0, 1, 1000).reshape(-1, 1)
f_data = 0.5 * x_data
combined_data = torch.column_stack((x_data, f_data))

X = pioneer.config.Variable("x", 1)
U = pioneer.config.Variable("u", 1)
F = pioneer.config.Variable("f", 1)

input_config = pioneer.config.DataConfiguration(
    pioneer.config.BatchAxes(1000), pioneer.config.FeatureAxes(X * F)
)

dataset = pioneer.data.ArrayLikeDataSet(data=[combined_data], data_configs=[input_config])

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

constraint = pioneer.constraints.MSEConstraint(
    model.output_ports[0].data_configuration,
)

computation_graph = pioneer.Graph()

computation_graph.connect(data_loader.get_output_port(X), model)
computation_graph.connect(model, constraint.input_1)
computation_graph.connect(data_loader.get_output_port(U), constraint.input_2)

computation_graph.setup()

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
