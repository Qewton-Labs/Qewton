import pickle

import torch

from qewton.optim.base import EvaluationPhase
from qewton.optim.trainer.training_controllers import TrainerState


def _state():
    state = TrainerState("results")
    state.current_optimization_phase = object()  # holds the optimizer in practice
    state.losses[EvaluationPhase.TRAIN]["loss"] = torch.tensor(0.5, requires_grad=True)
    state.history.append("entry")
    state.termination_reason = "Training finished"
    return state


def test_detached_copy_has_no_model_references():
    copied = _state().detached_copy()

    assert not hasattr(copied, "current_optimization_phase")
    assert copied.losses[EvaluationPhase.TRAIN]["loss"] == 0.5
    assert isinstance(copied.losses[EvaluationPhase.TRAIN]["loss"], float)
    assert copied.termination_reason == "Training finished"
    assert copied.history == ["entry"]


def test_detached_copy_leaves_the_state_unchanged():
    state = _state()
    phase = state.current_optimization_phase
    state.detached_copy()

    assert state.current_optimization_phase is phase
    assert torch.is_tensor(state.losses[EvaluationPhase.TRAIN]["loss"])


def test_detached_copy_can_be_pickled():
    copied = pickle.loads(pickle.dumps(_state().detached_copy()))
    assert copied.losses[EvaluationPhase.TRAIN]["loss"] == 0.5
