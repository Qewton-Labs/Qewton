import multiprocessing as mp
import torch
import pioneer


def build_problem():
    in_data = torch.load("/localdata/tomfre/FNO_data_Stokes/stokes_input.pt")
    out_data = torch.load("/localdata/tomfre/FNO_data_Stokes/stokes_output.pt")
    data = torch.cat((in_data, out_data), dim=-1)
    C = pioneer.config.Variable("chi", 1)
    U = pioneer.config.Variable("u", 2)

    dataset = pioneer.data.DataSet.from_data(data, C * U, batch_size=5000)
    model = pioneer.algorithms.TorchPCANN(
        C,
        U,
        dataset,
        dataset,
        input_pca_components=pioneer.optim.DiscreteHyperparameter((8, 100)),
        output_pca_components=pioneer.optim.DiscreteHyperparameter((8, 100)),
        hidden_layers=pioneer.optim.DiscreteHyperparameter((1, 4)),
        hidden_neurons=pioneer.optim.DiscreteHyperparameter((16, 100)),
        activation_fn=pioneer.optim.CategoricalHyperparameter(
            [torch.nn.Tanh(), torch.nn.ReLU()]
        ),
    )

    mse_constraint = pioneer.constraints.MSEConstraint(
        model[model.OutputKeys.OUTPUT].data_configuration, relative=True
    )

    pipeline = pioneer.pipelines.MSEDataPipeline(
        dataset, model, constraint=mse_constraint
    )

    trainer = pioneer.optim.trainer.PyTorchTrainer(
        [pipeline],
        [pipeline.mse_constraint],
        torch.optim.Adam,
        max_iterations=10000,
        learning_rate=0.001,
    )
    trainer.set_tuning_constraints([pipeline.mse_constraint])
    return trainer


# trainer.run()
if __name__ == "__main__":
    mp.set_start_method("spawn")
    tuner = pioneer.optim.tuner.GridSearchTuner(
        build_problem,
        trial_number=80,
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        trials_per_device=2,
        save_path="examples/pca_tuner_stokes_domain",
    )
    tuner.run()
