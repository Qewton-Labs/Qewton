import torch
import pioneer


def build_problem():
    x_data = torch.linspace(0, 1, 1000).reshape(-1, 1)
    u_data = x_data**2 + torch.sin(6.0 * x_data)

    X = pioneer.config.Variable("x", 1)
    U = pioneer.config.Variable("u", 1)
    dataset_X = pioneer.data.DataSet.from_data(x_data, X, batch_size=1000)
    dataset_U = pioneer.data.DataSet.from_data(u_data, U, batch_size=1000)

    model = pioneer.algorithms.FCN(
        in_neurons=1,
        n_hidden_layers=pioneer.optim.DiscreteHyperparameter((1, 3)),
        hidden_neurons=pioneer.optim.DiscreteHyperparameter((6, 32)),
        out_neurons=1,
        activation=pioneer.optim.CategoricalHyperparameter(
            [pioneer.building_blocks.Tanh, pioneer.building_blocks.ReLU]
        ),
    )

    constraint = pioneer.constraints.MSEConstraint(
        model.output_ports[0].data_configuration,
    )

    computation_graph = pioneer.Graph()

    computation_graph.connect(dataset_X, model)
    computation_graph.connect(model, constraint.input_1)
    computation_graph.connect(dataset_U, constraint.input_2)

    adam_phase = pioneer.optim.OptimizationPhase(
        optimizer=pioneer.optim.Adam(),
        lr=0.001,
        max_iterations=2000,
    )

    trainer = pioneer.optim.GraphBasedTrainer(
        optimization_phases=adam_phase,
        graphs=[computation_graph],
        training_constraints=[constraint],
        device="cpu",
    )
    trainer.set_tuning_constraints([constraint])

    return trainer


# trainer.run()
# tuner = pioneer.optim.tuner.GridSearchTuner(
#     build_problem,
#     trial_number=30,
#     devices=["cuda:1", "cuda:2", "cuda:3"],
#     trials_per_device=2,
# )
# tuner.run()
