import torch
import qewton

x_data = torch.linspace(0, 1, 1000).reshape(-1, 1)
u_data = x_data**2 + torch.sin(6.0 * x_data)
data = torch.column_stack((x_data, u_data))

X = qewton.config.Variable("x", 1)
U = qewton.config.Variable("u", 1)

input_config = qewton.config.DataConfiguration(
    qewton.config.BatchAxes(1000), qewton.config.FeatureAxes(X)
)
output_config = qewton.config.DataConfiguration(
    qewton.config.BatchAxes(1000), qewton.config.FeatureAxes(U)
)

dataset = qewton.data.ArrayLikeDataSet(
    data=[x_data, u_data], data_configs=[input_config, output_config]
)

data_loader = qewton.data.DataLoader(
    data_set=dataset,
    batch_size=1000,
    splitting_ratio=(1.0, 0.0, 0.0),
    shuffle_data=False,
)

model = qewton.algorithms.FCN(
    in_neurons=1,
    hidden_neurons=50,
    out_neurons=1,
    n_hidden_layers=6,
    activation=qewton.bb.Tanh,
)

constraint = qewton.constraints.MSEConstraint()
computation_graph = qewton.Graph()

computation_graph.connect(data_loader.get_output_port(X), model)
computation_graph.connect(model, constraint.input_1)
computation_graph.connect(data_loader.get_output_port(U), constraint.input_2)

constraint.name = "Constraint"
data_loader.name = "Input"
# end output ?
computation_graph.setup()


from qewton.visualization.graphs.base import GraphPlotter

plotter = GraphPlotter(computation_graph)
plotter.save_svg("computation_graph")


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
    graphs=[computation_graph],
    training_objectives=[constraint],
    device="cuda:0",
)

trainer.run()
