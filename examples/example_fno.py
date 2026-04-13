import multiprocessing as mp
import torch
import pioneer


def build_problem():
    in_data = torch.load("/localdata/tomfre/FNO_data_Stokes/stokes_input.pt")
    out_data = torch.load("/localdata/tomfre/FNO_data_Stokes/stokes_output.pt")
    data = torch.cat((in_data, out_data), dim=-1)
    C = pioneer.config.Variable("chi", 1)
    U = pioneer.config.Variable("u", 2)

    dataset = pioneer.data.DataSet.from_data(data, C * U, batch_size=2000)
    model = pioneer.algorithms.TorchPhysicsFNO(
        C,
        U,
        spatial_dimension=2,
        fourier_layers=2,
        hidden_channels=1,
        fourier_modes=(12, 12),
        skip_connections=True,
        linear_connections=True,
        positional_embedding=False,
    )

    mse_constraint = pioneer.constraints.MSEConstraint(
        model[model.OutputKeys.OUTPUT].data_configuration, relative=True
    )

    pipeline = pioneer.graphs.MSEDataPipeline(dataset, model, constraint=mse_constraint)

    trainer = pioneer.optim.trainer.PyTorchTrainer(
        [pipeline],
        [pipeline.mse_constraint],
        torch.optim.Adam,
        max_iterations=10000,
        learning_rate=0.001,
        device="cuda:0",
    )
    trainer.set_tuning_constraints([pipeline.mse_constraint])
    return trainer


problem_trainer = build_problem()
problem_trainer.run()
# if __name__ == "__main__":
#     mp.set_start_method("spawn")
#     tuner = pioneer.optim.tuner.GridSearchTuner(
#         build_problem,
#         trial_number=80,
#         devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
#         trials_per_device=2,
#         save_path="examples/pca_tuner_stokes_domain",
#     )
#     tuner.run()
