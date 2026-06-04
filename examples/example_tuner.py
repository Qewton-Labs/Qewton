import torch
import qewton

x_data = torch.linspace(0, 1, 1000).reshape(-1, 1)
u_data = x_data**2 + torch.sin(6.0 * x_data)
torch.cat((x_data, u_data), dim=1)

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
    batch_size=qewton.optim.DiscreteHyperparameter((100, 1000)),
    splitting_ratio=(0.9, 0.1, 0.0),
    shuffle_data=False,
)

model = qewton.algorithms.FCN(
    in_neurons=1,
    n_hidden_layers=qewton.optim.DiscreteHyperparameter((1, 3)),
    hidden_neurons=qewton.optim.DiscreteHyperparameter((6, 32)),
    out_neurons=1,
    activation=qewton.optim.CategoricalHyperparameter(
        [qewton.building_blocks.Tanh, qewton.building_blocks.ReLU]
    ),
)

constraint = qewton.constraints.MSEConstraint(
    evaluated_in_mode=qewton.optim.EvaluationPhase.TRAIN
)

computation_graph = qewton.Graph()

computation_graph.connect(data_loader.get_output_port(X), model)
computation_graph.connect(model, constraint.input_1)
computation_graph.connect(data_loader.get_output_port(U), constraint.input_2)

computation_graph.setup()

adam_phase = qewton.optim.OptimizationPhase(
    optimizer=qewton.optim.Adam(),
    lr=qewton.optim.ContinuousHyperparameter(
        (1e-4, 1e-2), scale=qewton.optim.HyperParameterScale.LOG
    ),
    max_iterations=qewton.optim.CategoricalHyperparameter([500, 1000, 2000]),
)

trainer = qewton.optim.GraphBasedTrainer(
    optimization_phases=adam_phase,
    graphs=[computation_graph],
    training_objectives=[constraint],
    callbacks=[qewton.optim.CSVLogger(log_phase=qewton.optim.EvaluationPhase.TRAIN)],
    device="cpu",
)

tuner = qewton.optim.tuner.GridSearchTuner(
    trainer,
    tuning_objectives=[constraint],
    trial_number=50,
    devices=["cuda:0", "cuda:1"],
    trials_per_device=2,
)
tuner.run()
