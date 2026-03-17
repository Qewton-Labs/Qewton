import numpy as np
import torch
import pioneer


def build_problem():
    x_data = np.linspace(0, 1, 1000).reshape(-1, 1)
    u_data = 100.0 * np.sin(6.0 * x_data)
    data = torch.tensor(np.column_stack((x_data, u_data)), dtype=torch.float32)

    X = pioneer.config.Variable("x", 1)
    U = pioneer.config.Variable("u", 1)

    dataset = pioneer.data.DataSet.from_data(
        data, X * U, batch_size=800, shuffle_data=True
    )

    hp_layers = pioneer.optim.DiscreteHyperparameter((1, 3), name="Hidden Layer")

    model = pioneer.algorithms.TorchFCN(
        X,
        U,
        hidden_layers=hp_layers,
        hidden_neurons=pioneer.optim.DiscreteHyperparameter(
            (1, 16), active_when=(hp_layers >= 2)
        ),
        activation_fn=pioneer.optim.CategoricalHyperparameter(
            [torch.nn.Tanh(), torch.nn.ReLU()]
        ),
    )

    pipeline = pioneer.pipelines.MSEDataPipeline(dataset, model)

    trainer = pioneer.optim.trainer.PyTorchTrainer(
        [pipeline],
        [pipeline.mse_constraint],
        torch.optim.Adam,
        max_iterations=pioneer.optim.CategoricalHyperparameter([1000, 2000, 5000]),
        learning_rate=0.001,
        device="cuda:0",
    )

    trainer.set_tuning_constraints([pipeline.mse_constraint])
    return trainer


# trainer.run()
tuner = pioneer.optim.tuner.RandomSearchTuner(
    build_problem,
    trial_number=16,
    devices=["cpu"],
    trials_per_device=2,
)
tuner.run()
