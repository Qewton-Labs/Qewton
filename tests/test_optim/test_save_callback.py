import json
import os
from unittest.mock import patch

import pytest

try:
    import torch
except ImportError:
    torch = None

import qewton
from qewton.algorithms import FCN
from qewton.config import Variable, DataConfiguration, BatchAxes, FeatureAxes
from qewton.constraints import MSEConstraint
from qewton.constraints.base import ConstraintObjective
from qewton.data import ArrayLikeDataSet, DataLoader
from qewton.optim import OptimizationPhase, Adam, GraphBasedTrainer, EvaluationPhase
from qewton.optim.trainer import SaveBestModelCallback, TrainerState

SAVE_FN = "qewton.optim.trainer.callbacks.save_callback.save"


def _model(name="fcn"):
    model = FCN(in_neurons=1, hidden_neurons=2, out_neurons=1, n_hidden_layers=1)
    model.name = name
    return model


def _run_steps(callback, state, constraint, values, phase=EvaluationPhase.TRAIN):
    # Feeds one loss value per step and returns the steps where a save happened.
    saved_at = []
    with patch(SAVE_FN) as save_mock:
        for step, value in enumerate(values):
            state.losses[phase][constraint.name] = value
            state.iteration += 1
            calls_before = save_mock.call_count
            callback.on_train_step_end(state)
            if save_mock.call_count > calls_before:
                saved_at.append(step)
    return saved_at


@pytest.fixture
def setup(tmp_path):
    constraint = MSEConstraint()
    state = TrainerState(str(tmp_path))
    return constraint, state


def test_saves_only_on_improvement(setup):
    constraint, state = setup
    cb = SaveBestModelCallback(_model(), constraint, check_interval=1)
    cb.on_training_start(state)
    cb.on_optimization_phases_start(state)

    saved_at = _run_steps(cb, state, constraint, [1.0, 0.5, 0.7, 0.3, float("nan"), 0.3])

    assert saved_at == [0, 1, 3]
    assert cb.best_value == 0.3
    assert cb.best_iteration == 4


def test_check_interval_counts_per_phase(setup):
    constraint, state = setup
    cb = SaveBestModelCallback(_model(), constraint, check_interval=3)
    cb.on_training_start(state)

    cb.on_optimization_phases_start(state)
    first = _run_steps(cb, state, constraint, [5.0, 4.0, 3.0, 2.0, 1.0])
    cb.on_optimization_phases_start(state)
    second = _run_steps(cb, state, constraint, [0.5, 0.1])

    assert first == [0, 3]
    assert second == [0]  # step counter restarts with the new phase


def test_skips_check_while_loss_missing(setup):
    constraint, state = setup
    cb = SaveBestModelCallback(_model(), constraint, check_interval=1)
    cb.on_training_start(state)
    cb.on_optimization_phases_start(state)

    assert _run_steps(cb, state, constraint, [None, 2.0]) == [1]


def test_maximize_objective(setup):
    constraint, state = setup
    constraint.objective = ConstraintObjective.MAXIMIZE
    cb = SaveBestModelCallback(_model(), constraint, check_interval=1)
    cb.on_training_start(state)
    cb.on_optimization_phases_start(state)

    assert _run_steps(cb, state, constraint, [0.1, 0.5, 0.2, 0.9]) == [0, 1, 3]


def test_multiple_constraints_are_summed(setup):
    c1, state = setup
    c2 = MSEConstraint()
    c2.name = "second"
    cb = SaveBestModelCallback(_model(), [c1, c2], check_interval=1)
    cb.on_training_start(state)
    cb.on_optimization_phases_start(state)

    saved = []
    with patch(SAVE_FN) as save_mock:
        for step, (v1, v2) in enumerate([(1.0, 1.0), (0.5, 2.0), (0.5, 1.0)]):
            state.losses[EvaluationPhase.TRAIN][c1.name] = v1
            state.losses[EvaluationPhase.TRAIN][c2.name] = v2
            before = save_mock.call_count
            cb.on_train_step_end(state)
            if save_mock.call_count > before:
                saved.append(step)

    assert saved == [0, 2]  # sums 2.0, 2.5, 1.5
    assert cb.best_value == 1.5


def test_prefers_validation_loss(setup):
    constraint, state = setup
    state.losses[EvaluationPhase.TRAIN][constraint.name] = 100.0
    cb = SaveBestModelCallback(_model(), constraint, check_interval=1)
    cb.on_training_start(state)
    cb.on_optimization_phases_start(state)

    _run_steps(cb, state, constraint, [3.0], phase=EvaluationPhase.VALIDATION)
    assert cb.best_value == 3.0


def test_explicit_phase(setup):
    constraint, state = setup
    state.losses[EvaluationPhase.VALIDATION][constraint.name] = 100.0
    cb = SaveBestModelCallback(
        _model(), constraint, check_interval=1, evaluation_phase=EvaluationPhase.TRAIN
    )
    cb.on_training_start(state)
    cb.on_optimization_phases_start(state)

    _run_steps(cb, state, constraint, [3.0])
    assert cb.best_value == 3.0


def test_unknown_constraint_raises(setup):
    constraint, state = setup
    cb = SaveBestModelCallback(_model(), constraint, check_interval=1)
    cb.on_training_start(state)
    cb.on_optimization_phases_start(state)
    with pytest.raises(KeyError, match="MSEConstraint"):
        cb.on_train_step_end(state)


def test_duplicate_node_names_raise(setup):
    constraint, state = setup
    cb = SaveBestModelCallback([_model("a"), _model("a")], constraint)
    with pytest.raises(ValueError, match="unique names"):
        cb.on_training_start(state)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        SaveBestModelCallback([], MSEConstraint())
    with pytest.raises(ValueError):
        SaveBestModelCallback(_model(), [])
    with pytest.raises(ValueError):
        SaveBestModelCallback(_model(), MSEConstraint(), check_interval=0)


@pytest.mark.skipif(torch is None, reason="PyTorch is required for training.")
def test_saves_loadable_model_during_training(tmp_path):
    torch.set_num_threads(1)
    x_var, u_var = Variable("x", 1), Variable("u", 1)
    x_data = torch.linspace(0, 1, 20).reshape(-1, 1)
    dataset = ArrayLikeDataSet(
        data=[x_data, x_data**2],
        data_configs=[
            DataConfiguration(BatchAxes(20), FeatureAxes(x_var)),
            DataConfiguration(BatchAxes(20), FeatureAxes(u_var)),
        ],
    )
    data_loader = DataLoader(dataset, batch_size=20)
    model = _model()
    constraint = MSEConstraint()

    graph = qewton.Graph()
    graph.connect(data_loader.get_output_port(x_var), model)
    graph.connect(model, constraint.input_1)
    graph.connect(data_loader.get_output_port(u_var), constraint.input_2)

    callback = SaveBestModelCallback(model, constraint, check_interval=5)
    trainer = GraphBasedTrainer(
        optimization_phases=[OptimizationPhase(optimizer=Adam(), lr=0.01, max_iterations=20)],
        graphs=[graph],
        training_objectives=[constraint],
        callbacks=[callback],
        save_path=str(tmp_path / "run"),
    )
    trainer.run(show_progress=False)

    folder = os.path.join(trainer.train_state.save_path, "best_model")
    with open(os.path.join(folder, "checkpoint_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    assert info["iteration"] == callback.best_iteration > 0
    assert info["value"] == callback.best_value
    assert info["constraints"] == [constraint.name]

    loaded = qewton.load(os.path.join(folder, model.name))
    assert isinstance(loaded, FCN)
