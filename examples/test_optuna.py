import multiprocessing as mp
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import numpy as np
import torch


import pioneer


def build_problem():
    x_data = np.linspace(0, 1, 1000).reshape(-1, 1)
    u_data = 100.0 * np.sin(6.0 * x_data) + 30.0 * np.cos(20.0 * x_data)
    data = torch.tensor(np.column_stack((x_data, u_data)), dtype=torch.float32)

    X = pioneer.config.Variable("x", 1)
    U = pioneer.config.Variable("u", 1)

    dataset = pioneer.data.DataSet.from_data(
        data, X * U, batch_size=800, shuffle_data=True
    )

    model = pioneer.algorithms.TorchFCN(
        X,
        U,
        hidden_layers=pioneer.optim.DiscreteHyperparameter((1, 5)),
        hidden_neurons=pioneer.optim.DiscreteHyperparameter((1, 32)),
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
        learning_rate=pioneer.optim.ContinuousHyperparameter(
            (0.001, 0.1), scale=pioneer.optim.HyperParameterScale.LOG
        ),
        device="cuda:0",
    )

    trainer.set_tuning_constraints([pipeline.mse_constraint])
    return trainer


if __name__ == "__main__":
    mp.set_start_method("spawn")
    study = optuna.create_study(
        study_name="journal_storage_multiprocess",
        storage="sqlite:///examples/test_optuna/test_study.db",
        # JournalStorage(JournalFileBackend(file_path="./test.log")),
        load_if_exists=True,  # Useful for multi-process or multi-node optimization.
    )
    tuner = pioneer.optim.tuner.OptunaTuner(
        build_problem,
        optuna_study=study,
        trial_number=40,
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        trials_per_device=1,
        save_path="examples/test_optuna",
    )
    tuner.run()
