import torch
import optuna
import pioneer


x_data = torch.linspace(0, 1, 1000).reshape(-1, 1)
u_data = x_data**2 + torch.sin(6.0 * x_data)

X = pioneer.config.Variable("x", 1)
U = pioneer.config.Variable("u", 1)

input_config = pioneer.config.DataConfiguration(
    pioneer.config.BatchAxes(1000), pioneer.config.FeatureAxes(X)
)
output_config = pioneer.config.DataConfiguration(
    pioneer.config.BatchAxes(1000), pioneer.config.FeatureAxes(U)
)

dataset = pioneer.data.ArrayLikeDataSet(
    data=[x_data, u_data], data_configs=[input_config, output_config]
)

data_loader = pioneer.data.DataLoader(
    data_set=dataset,
    batch_size=pioneer.optim.DiscreteHyperparameter((100, 1000)),
    splitting_ratio=(0.9, 0.1, 0.0),
    shuffle_data=False,
)

model = pioneer.algorithms.FCN(
    in_neurons=1,
    n_hidden_layers=pioneer.optim.DiscreteHyperparameter((1, 3)),
    hidden_neurons=pioneer.optim.DiscreteHyperparameter((6, 32)),
    out_neurons=1,
    activation=pioneer.optim.CategoricalHyperparameter(
        [pioneer.building_blocks.Tanh, pioneer.building_blocks.ReLU]
    ),
)

constraint = pioneer.constraints.MSEConstraint()

computation_graph = pioneer.Graph()

computation_graph.connect(data_loader.get_output_port(X), model)
computation_graph.connect(model, constraint.input_1)
computation_graph.connect(data_loader.get_output_port(U), constraint.input_2)

computation_graph.setup()

adam_phase = pioneer.optim.OptimizationPhase(
    optimizer=pioneer.optim.Adam(),
    lr=0.001,
    max_iterations=pioneer.optim.CategoricalHyperparameter([500, 1000, 2000]),
)

trainer = pioneer.optim.GraphBasedTrainer(
    optimization_phases=adam_phase,
    graphs=[computation_graph],
    training_objectives=[constraint],
    device="cpu",
)

study = optuna.create_study(
    study_name="journal_storage_multiprocess",
    storage="sqlite:///examples/test_optuna/test_study.db",
    # JournalStorage(JournalFileBackend(file_path="./test.log")),
    load_if_exists=True,  # Useful for multi-process or multi-node optimization.
)
tuner = pioneer.optim.tuner.OptunaTuner(
    trainer,
    tuning_objectives=[constraint],
    optuna_study=study,
    trial_number=100,
    devices=["cuda:0", "cuda:1"],
    trials_per_device=4,
    save_path="examples/test_optuna",
)
tuner.run()
