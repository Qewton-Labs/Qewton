import math
import qewton
import torch

N_batch = 1000
N_x = 100

x_data = torch.linspace(0, 1, N_x).reshape(-1, 1)
u_data = torch.zeros((N_batch, N_x, 1))
f_data = torch.zeros((N_batch, N_x, 1))
for i in range(N_batch):
    function_type = torch.randint(0, 3, (1,))
    if function_type == 0:
        f_data[i] = torch.randn((1,)) * x_data
        u_data[i] = f_data[i] / 2.0 * x_data
    if function_type == 1:
        a, b = torch.randn((1,)), torch.randn((1,))
        f_data[i] = a * torch.sin(2.0 * math.pi * (x_data + b))
        u_data[i] = -a / (2.0 * math.pi) * torch.cos(2.0 * math.pi * (x_data + b))
    else:
        a, b, c = torch.randn((1,)), torch.randn((1,)), torch.randn((1,))
        f_data[i] = a * torch.cos(2.0 * math.pi * (x_data + b)) + c * x_data**2
        u_data[i] = (
            -a / (2.0 * math.pi) * torch.cos(2.0 * math.pi * (x_data + b))
            + c / 3.0 * x_data**3
        )


F = qewton.config.Variable("f", 1)
U = qewton.config.Variable("u", 1)


f_config = qewton.config.DataConfiguration(
    qewton.config.BatchAxes(N_batch),
    qewton.config.GeometryAxes(shape=(N_x,)),
    qewton.config.FeatureAxes(F),
)
u_config = qewton.config.DataConfiguration(
    qewton.config.BatchAxes(N_batch),
    qewton.config.GeometryAxes(shape=(N_x,)),
    qewton.config.FeatureAxes(U),
)


dataset = qewton.data.ArrayLikeDataSet(
    data=[f_data, u_data], data_configs=[f_config, u_config]
)

data_loader = qewton.data.DataLoader(
    data_set=dataset,
    batch_size=400,
    splitting_ratio=(0.8, 0.0, 0.0),
    shuffle_data=False,
)

pca_n_f = qewton.DiscreteHyperparameter([10, 50])
pca_n_u = qewton.DiscreteHyperparameter([10, 50])

model = qewton.algorithms.FCN(
    in_neurons=pca_n_f,
    hidden_neurons=50,
    out_neurons=pca_n_u,
    n_hidden_layers=2,
    activation=qewton.bb.Tanh,
)
constraint = qewton.constraints.MSEConstraint()
pca_node_f = qewton.PCANode(n=pca_n_f, data_source_node=data_loader)
pca_node_u = qewton.PCANode(n=pca_n_u, data_source_node=data_loader)
inverse_pca_node_u = qewton.InversePCANode(pca_node=pca_node_u)

computation_graph = qewton.Graph()
computation_graph.connect(data_loader.get_output_port(F), pca_node_f)
computation_graph.connect(pca_node_f.output, model)
computation_graph.connect(model, inverse_pca_node_u)
computation_graph.connect(inverse_pca_node_u, constraint.input_1)
computation_graph.connect(data_loader.get_output_port(U), pca_node_u)
computation_graph.connect(data_loader.get_output_port(U), constraint.input_2)


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
