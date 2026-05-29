from typing import Callable

from ....base import EvaluationPhase

# File contains how each backend should setup the optimizers and do the optimization step.
# This is then used in the training controller to do the optimization step for each
# backend in a unified way.


# TODO: Make consistent with backend class structure
def _pytorch_setup_optimizer(optimization_phase, trainer):
    torch_params = []
    for param in trainer.trainable_parameters:
        torch_params.append(param.parameters)

    optimizer_obj = optimization_phase.optimizer(
        torch_params,  # type: ignore
        optimization_phase.lr.value,
        **optimization_phase.optimizer_args,
    )
    return optimizer_obj


def _pytorch_do_optimization_step(
    optimization_phase,
    eval_function: Callable,
    step_idx: int,
    train_state,
):
    if optimization_phase.optimizer.requires_closure:
        optimization_phase.optimizer_obj.step(
            lambda: _pytorch_closure(
                optimization_phase, eval_function, step_idx, train_state
            )
        )
        total_loss = _pytorch_closure(
            optimization_phase, eval_function, step_idx, train_state
        )
    else:
        total_loss = _pytorch_closure(
            optimization_phase, eval_function, step_idx, train_state
        )
        optimization_phase.optimizer_obj.step()

    train_state.total_train_loss = total_loss.item()  # type: ignore


def _pytorch_closure(
    optimization_phase,
    eval_function: Callable,
    step_idx: int,
    train_state,
):
    optimization_phase.optimizer_obj.zero_grad()
    eval_function(step_idx)

    total_loss = 0.0
    for loss_value in train_state.losses[EvaluationPhase.TRAIN].values():
        total_loss += loss_value

    total_loss.backward()  # type: ignore
    return total_loss


def _pytorch_cleanup(backend):
    backend.library.cuda.empty_cache()
    backend.library.cuda.ipc_collect()
