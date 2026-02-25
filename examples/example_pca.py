import torch
import pioneer


def build_problem():
    in_data = torch.load("/localdata/tomfre/PCA_data_Stokes/stokes_input.pt")
    out_data = torch.load("/localdata/tomfre/PCA_data_Stokes/stokes_output.pt")

    F = pioneer.config.Variable("f", 2)
    U = pioneer.config.Variable("u", 2)

    dataset_in = pioneer.data.DataSet.from_data(
        in_data, F, batch_size=5000, name="Input Data", shuffle_data=False
    )
    dataset_out = pioneer.data.DataSet.from_data(
        out_data, U, batch_size=5000, name="Output Data", shuffle_data=False
    )

    model = pioneer.algorithms.TorchPCANN(
        F,
        U,
        dataset_in,
        dataset_out,
        input_pca_components=pioneer.optim.DiscreteHyperparameter(
            (8, 50), initial_value=10
        ),
        output_pca_components=pioneer.optim.DiscreteHyperparameter(
            (8, 100), initial_value=20
        ),
        hidden_layers=pioneer.optim.DiscreteHyperparameter((1, 4), initial_value=2),
        hidden_neurons=pioneer.optim.DiscreteHyperparameter((16, 100), initial_value=32),
        activation_fn=pioneer.optim.CategoricalHyperparameter(
            [torch.nn.Tanh(), torch.nn.ReLU()]
        ),
    )

    constraint = pioneer.constraints.MSEConstraint(
        model[model.OutputKeys.OUTPUT].data_configuration,
    )

    pipeline = pioneer.pipelines.Pipeline()

    pipeline.connect(dataset_in, model)
    pipeline.connect(model, constraint[constraint.InputKeys.INPUT1])
    pipeline.connect(dataset_out, constraint[constraint.InputKeys.INPUT2])

    pipeline.validate()

    trainer = pioneer.optim.trainer.PyTorchTrainer(
        [pipeline],
        [constraint],
        torch.optim.Adam,
        max_iterations=pioneer.optim.CategoricalHyperparameter(
            [1000, 2000, 5000], initial_value=5000
        ),
        learning_rate=0.001,
        device="cuda:0",
    )

    trainer.set_tuning_constraints([constraint])
    return trainer


tuner = pioneer.optim.tuner.GridSearchTuner(
    build_problem,
    trial_number=100,
    devices=["cuda:1", "cuda:2", "cuda:3", "cuda:0"],
    trials_per_device=2,
    save_path="examples/pca_tuner_stokes",
)
tuner.run()
