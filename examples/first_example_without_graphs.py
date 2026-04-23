import torch
import pioneer

device = "cuda:0"
x_data = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
u_data = x_data**2 + torch.sin(6.0 * x_data)

model = pioneer.algorithms.FCN(
    in_neurons=1,
    hidden_neurons=50,
    out_neurons=1,
    n_hidden_layers=1,
    activation=pioneer.building_blocks.Tanh,
)


def mse_loss_fn(_iteration_idx, _train_state):
    model_out = model(x_data)
    loss = torch.mean((model_out - u_data) ** 2)
    return loss


adam_phase = pioneer.optim.OptimizationPhase(
    optimizer=pioneer.optim.Adam(),
    lr=0.001,
    max_iterations=2000,
)

lbfgs_phase = pioneer.optim.OptimizationPhase(
    optimizer=pioneer.optim.LBFGS(),
    lr=0.1,
    max_iterations=50,
    optimizer_args={"max_eval": 10},
)

trainer = pioneer.optim.FunctionBasedTrainer(
    optimization_phases=[adam_phase, lbfgs_phase],
    training_functions=[mse_loss_fn],
    model_nodes=[model],
    device=device,
)
trainer.run()
