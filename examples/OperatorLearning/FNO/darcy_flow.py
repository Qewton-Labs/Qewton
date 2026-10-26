"""Compares Qewton's FNO with the FNO of neuralop on the small Darcy-flow dataset.

First, the outputs of both FNOs are compared for the same weights and a range of
options. Then both models, with the same architecture and the same initial weights, are
trained on the same batches with the same settings, so their results should agree.
Requires the `neuraloperator` package and an internet connection for the dataset.
"""

import time

import neuralop
import torch
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params

import qewton
from qewton.algorithms.building_blocks.conv import Flattened1dConv
from qewton.algorithms.building_blocks.skip_connections import SoftGating
from qewton.algorithms.dl_models.fno import FNO

# %%
SEED = 0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

N_TRAIN = 1000
BATCH_SIZE = 64
TEST_RESOLUTIONS = [16, 32]
N_TESTS = [100, 50]
TEST_BATCH_SIZES = [32, 32]

N_MODES = (8, 8)
HIDDEN_CHANNELS = 24
N_LAYERS = 4
LIFTING_CHANNEL_RATIO = 2
PROJECTION_CHANNEL_RATIO = 2

# Settings of neuralop's Darcy-flow example, except for the MSE instead of the H1 loss
# and Adam instead of AdamW (neuralop's AdamW updates complex weights differently).
N_EPOCHS = 15
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
SCHEDULER_T_MAX = 30

L2_LOSS = neuralop.LpLoss(d=2, p=2)


def mse_loss(output: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
    """Mean squared error, summed over the last axis and averaged over all others.

    Same reduction as `qewton.constraints.MSEConstraint`. The target is called `y` and
    extra keyword arguments are accepted because neuralop's `Trainer` calls its losses
    as `loss(output, **batch)`.
    """
    return ((output - y) ** 2).sum(dim=-1).mean()


def skip_parameters(prefix: str, skip: qewton.Node | None) -> dict[str, torch.Tensor]:
    """Maps the weight of a skip connection to neuralop's parameter name."""
    if isinstance(skip, Flattened1dConv):
        return {f"{prefix}.conv.weight": skip.conv.kernel._trainable_parameter}
    if isinstance(skip, SoftGating):
        return {f"{prefix}.weight": skip.weight._trainable_parameter}
    return {}


def channel_mlp_parameters(prefix: str, channel_mlp) -> dict[str, torch.Tensor]:
    """Maps the weights of a channel MLP to neuralop's parameter names."""
    parameters = {}
    for i, layer in enumerate(channel_mlp._graph.sorted_nodes[::2]):
        parameters[f"{prefix}.fcs.{i}.weight"] = layer.conv.kernel._trainable_parameter
        parameters[f"{prefix}.fcs.{i}.bias"] = layer.conv.bias._trainable_parameter
    return parameters


def qewton_parameter_map(model: FNO) -> dict[str, torch.Tensor]:
    """Maps the parameter names of neuralop's FNO to the tensors of a Qewton FNO.

    Relies on the structure of the Qewton FNO: optional grid embedding, lifting, Fourier
    layers and projection, in this order.
    """
    nodes = model._graph.sorted_nodes
    if model.positional_embedding.value is not None:
        nodes = nodes[1:]
    lifting, *fourier_layers, projection = nodes

    parameters = {
        **channel_mlp_parameters("lifting", lifting),
        **channel_mlp_parameters("projection", projection),
    }
    blocks = "fno_blocks"
    for i, layer in enumerate(fourier_layers):
        parameters[f"{blocks}.convs.{i}.weight.tensor"] = (
            layer.conv.weight._trainable_parameter
        )
        parameters[f"{blocks}.convs.{i}.bias"] = layer.conv.bias._trainable_parameter
        parameters.update(
            skip_parameters(f"{blocks}.fno_skips.{i}", layer.fno_skip_connection)
        )
        if layer.channel_mlp is not None:
            parameters.update(
                skip_parameters(
                    f"{blocks}.channel_mlp_skips.{i}", layer.channel_mlp_skip_connection
                )
            )
            parameters.update(
                channel_mlp_parameters(f"{blocks}.channel_mlp.{i}", layer.channel_mlp)
            )
    return parameters


def copy_weights(model: FNO, source: neuralop.models.FNO) -> None:
    """Copies the weights of a neuralop FNO into a Qewton FNO."""
    weights = source.state_dict()
    parameters = qewton_parameter_map(model)
    weights = {name: w for name, w in weights.items() if name != "_metadata"}
    assert set(parameters) == set(weights), "The parameters of the models differ."
    with torch.no_grad():
        for name, tensor in parameters.items():
            tensor.copy_(weights[name])


# %% Output check: with the same weights, both FNOs must compute the same output
OUTPUT_CHECK_TOLERANCE = 1e-5
OUTPUT_CHECK_ARCHITECTURE = dict(in_channels=2, out_channels=3, hidden_channels=8)
# (options of both FNOs, shape of the input without batch and channel axes)
OUTPUT_CHECK_CASES = [
    (dict(n_modes=(6, 6), n_layers=3), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=1), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=3, use_channel_mlp=False), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=2, use_channel_mlp=False, fno_skip=None), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=2, fno_skip="identity"), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=2, fno_skip="soft-gating"), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=2, channel_mlp_skip="linear"), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=2, channel_mlp_skip="identity"), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=2, channel_mlp_skip=None), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=2, channel_mlp_expansion=1.0), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=2, lifting_channel_ratio=0), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=2, positional_embedding=None), (16, 16)),
    (dict(n_modes=(6, 6), n_layers=2), (16, 15)),  # odd size of the last dimension
    (dict(n_modes=(10, 10), n_layers=2), (8, 8)),  # more modes than frequencies
    (dict(n_modes=(6,), n_layers=2), (16,)),
    (dict(n_modes=(4, 4, 4), n_layers=2), (8, 8, 8)),
]


def check_outputs() -> None:
    """Checks for every case that both FNOs give the same output for the same weights."""
    for options, input_shape in OUTPUT_CHECK_CASES:
        options = {**OUTPUT_CHECK_ARCHITECTURE, **options}
        neuralop_fno = neuralop.models.FNO(**options)
        qewton_fno = FNO(**options)
        copy_weights(qewton_fno, neuralop_fno)
        x = torch.randn(3, options["in_channels"], *input_shape)
        with torch.no_grad():
            max_difference = (qewton_fno(x) - neuralop_fno(x)).abs().max().item()
        shown_options = {
            k: v for k, v in options.items() if k not in OUTPUT_CHECK_ARCHITECTURE
        }
        print(f"max output difference {max_difference:.1e} for {shown_options}")
        assert max_difference < OUTPUT_CHECK_TOLERANCE


torch.manual_seed(SEED)
check_outputs()


# %% Data
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=N_TRAIN,
    batch_size=BATCH_SIZE,
    n_tests=N_TESTS,
    test_resolutions=TEST_RESOLUTIONS,
    test_batch_sizes=TEST_BATCH_SIZES,
)
data_processor = data_processor.to(DEVICE)

# neuralop trains on normalized targets, so Qewton gets the same ones
train_inputs = train_loader.dataset.x
train_targets = data_processor.out_normalizer.transform(train_loader.dataset.y)

# %% Models
neuralop_model = neuralop.models.FNO(
    n_modes=N_MODES,
    in_channels=1,
    out_channels=1,
    hidden_channels=HIDDEN_CHANNELS,
    n_layers=N_LAYERS,
    lifting_channel_ratio=LIFTING_CHANNEL_RATIO,
    projection_channel_ratio=PROJECTION_CHANNEL_RATIO,
).to(DEVICE)

qewton_model = FNO(
    n_modes=N_MODES,
    in_channels=1,
    out_channels=1,
    hidden_channels=HIDDEN_CHANNELS,
    n_layers=N_LAYERS,
    lifting_channel_ratio=LIFTING_CHANNEL_RATIO,
    projection_channel_ratio=PROJECTION_CHANNEL_RATIO,
)


class CopyWeightsCallback(qewton.optim.Callback):
    """Copies the weights of a neuralop model into the Qewton model.

    The trainer re-initializes the network when training starts, so the weights have
    to be copied afterwards. Then it checks that both models compute the same output.
    """

    def __init__(self, model: FNO, source: neuralop.models.FNO, check_batch) -> None:
        super().__init__()
        self.model = model
        self.source = source
        self.check_batch = check_batch.to(DEVICE)

    def on_training_start(self, state: qewton.optim.TrainerState) -> None:
        copy_weights(self.model, self.source)
        with torch.no_grad():
            output_difference = self.model(self.check_batch) - self.source(
                self.check_batch
            )
        max_difference = output_difference.abs().max()
        print(f"Initial weights copied; max output difference: {max_difference:.1e}")
        assert max_difference < OUTPUT_CHECK_TOLERANCE


class CosineAnnealingCallback(qewton.optim.Callback):
    """Steps a cosine annealing schedule once per epoch."""

    def __init__(self, steps_per_epoch: int, t_max: int) -> None:
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.t_max = t_max
        self.scheduler = None

    def on_optimization_phases_start(self, state: qewton.optim.TrainerState) -> None:
        optimizer = state.current_optimization_phase.optimizer_obj
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max
        )

    def on_train_step_end(self, state: qewton.optim.TrainerState) -> None:
        # the iteration counter is already advanced here
        if state.iteration % self.steps_per_epoch == 0:
            self.scheduler.step()


# %% Train Qewton (first, so that the callback can copy the untrained neuralop weights)
PERMEABILITY = qewton.config.Variable("a", 1)
PRESSURE = qewton.config.Variable("u", 1)


def data_config(variable: qewton.config.Variable) -> qewton.config.DataConfiguration:
    return qewton.config.DataConfiguration(
        qewton.config.BatchAxes(len(train_inputs)),
        qewton.config.FeatureAxes(variable),
        qewton.config.GeometryAxes(shape=tuple(train_inputs.shape[2:])),
    )


dataset = qewton.data.ArrayLikeDataSet(
    data=[train_inputs, train_targets],
    data_configs=[data_config(PERMEABILITY), data_config(PRESSURE)],
)
# no shuffling: neuralop's train loader is sequential, too
data_loader = qewton.data.DataLoader(
    data_set=dataset,
    batch_size=BATCH_SIZE,
    splitting_ratio=(1.0, 0.0, 0.0),
    shuffle_data=False,
)

mse_constraint = qewton.constraints.MSEConstraint()
computation_graph = qewton.Graph()
computation_graph.connect(
    data_loader.get_output_port(PERMEABILITY), qewton_model.input_ports[0]
)
computation_graph.connect(qewton_model, mse_constraint.input_1)
computation_graph.connect(data_loader.get_output_port(PRESSURE), mse_constraint.input_2)

qewton_trainer = qewton.optim.GraphBasedTrainer(
    optimization_phases=[
        qewton.optim.OptimizationPhase(
            optimizer=qewton.optim.Adam(),
            lr=LEARNING_RATE,
            max_iterations=N_EPOCHS * data_loader.training_batches,
            optimizer_args={"weight_decay": WEIGHT_DECAY},
        )
    ],
    graphs=[computation_graph],
    training_objectives=[mse_constraint],
    callbacks=[
        CopyWeightsCallback(qewton_model, neuralop_model, train_inputs[:8]),
        CosineAnnealingCallback(data_loader.training_batches, SCHEDULER_T_MAX),
    ],
    device=DEVICE,
)
start_time = time.perf_counter()
qewton_trainer.run()
qewton_train_time = time.perf_counter() - start_time

# %% Train neuralop
optimizer = torch.optim.Adam(
    neuralop_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
neuralop_trainer = neuralop.Trainer(
    model=neuralop_model,
    n_epochs=N_EPOCHS,
    device=DEVICE,
    data_processor=data_processor,
    wandb_log=False,
    eval_interval=5,
    use_distributed=False,
    verbose=True,
)
start_time = time.perf_counter()
neuralop_trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX
    ),
    regularizer=False,
    training_loss=mse_loss,
    eval_losses={"mse": mse_loss, "l2": L2_LOSS},
)
neuralop_train_time = time.perf_counter() - start_time


# %% Comparison
def count_qewton_parameters(model: FNO) -> int:
    """Number of parameters, counted like neuralop does (complex values count twice)."""
    return sum(
        parameter.parameters.numel() * (2 if parameter.parameters.is_complex() else 1)
        for parameter in model.trainable_parameters
    )


@torch.no_grad()
def evaluate(model, loader) -> tuple[float, float]:
    """Mean MSE and mean relative L2 error of a model on decoded test outputs."""
    mse, l2, n_samples = 0.0, 0.0, 0
    for sample in loader:
        inputs, targets = sample["x"].to(DEVICE), sample["y"].to(DEVICE)
        outputs = data_processor.out_normalizer.inverse_transform(model(inputs))
        mse += mse_loss(outputs, targets).item() * len(inputs)
        l2 += L2_LOSS(outputs, targets).item()  # sum over the batch
        n_samples += len(inputs)
    return mse / n_samples, l2 / n_samples


neuralop_model.eval()
results = {
    "neuralop": (
        neuralop_model,
        count_model_params(neuralop_model),
        neuralop_train_time,
    ),
    "qewton": (qewton_model, count_qewton_parameters(qewton_model), qewton_train_time),
}
print(f"{'':10}{'params':>8}{'time [s]':>10}", end="")
for resolution in test_loaders:
    print(f"{f'mse@{resolution}':>12}{f'l2@{resolution}':>10}", end="")
print()
for name, (model, n_parameters, train_time) in results.items():
    print(f"{name:10}{n_parameters:>8}{train_time:>10.1f}", end="")
    for loader in test_loaders.values():
        test_mse, test_l2 = evaluate(model, loader)
        print(f"{test_mse:>12.4e}{test_l2:>10.4f}", end="")
    print()
