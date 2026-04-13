import multiprocessing as mp
from typing import Sequence
import numpy as np
import torch
import pioneer


class SimpleFCN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[torch.nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(prev_dim, output_dim))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_problem():
    x_data = np.linspace(0, 1, 1000).reshape(-1, 1)
    u_data = x_data**2 + np.sin(6.0 * x_data)
    data = torch.tensor(np.column_stack((x_data, u_data)), dtype=torch.float32)

    X = pioneer.config.Variable("x", 1)
    U = pioneer.config.Variable("u", 1)
    dataset = pioneer.nodes.DataSet.from_data(
        data, X * U, batch_size=1000, shuffle_data=True
    )

    model = pioneer.algorithms.PyTorchWrapper(
        dataset.data_config[X],
        dataset.data_config[U],
        model_cls=SimpleFCN,
        name="TestModel",
        input_dim=X.dim,
        hidden_dims=pioneer.optim.CategoricalHyperparameter(
            [[8, 8, 8], [8, 16], [8, 9], [3]]
        ),
        output_dim=U.dim,
    )

    slice_node = pioneer.nodes.SplitNode(dataset.data_config)

    constraint = pioneer.constraints.MSEConstraint(
        model[model.OutputKeys.OUTPUT].data_configuration,
    )

    pipeline = pioneer.graphs.Pipeline()

    pipeline.connect(dataset, slice_node)
    pipeline.connect(slice_node[X], model)
    pipeline.connect(slice_node[U], constraint[constraint.InputKeys.INPUT1])
    pipeline.connect(model, constraint[constraint.InputKeys.INPUT2])

    pipeline.validate()
    pipeline.visualize()
    # runtime = pipeline.create_runtime()
    # runtime.run()

    trainer = pioneer.optim.trainer.PyTorchTrainer(
        [pipeline],
        training_constraints=[constraint],
        optimizer=torch.optim.Adam,
        max_iterations=5000,
        learning_rate=pioneer.optim.ContinuousHyperparameter(
            (0.001, 0.1), scale=pioneer.optim.HyperParameterScale.LOG
        ),
        device="cpu",
    )
    # trainer.run()
    trainer.set_tuning_constraints([constraint])
    return trainer


# trainer.run()
if __name__ == "__main__":
    mp.set_start_method("spawn")
    tuner = pioneer.optim.tuner.GridSearchTuner(
        build_problem,
        trial_number=16,
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        trials_per_device=2,
        save_path="examples/wrapper",
    )
    tuner.run()
