import torch

from typing import Callable
from qewton.backends.optim import OptimBackend


class TorchOptimBackend(OptimBackend[torch.Tensor]):
    """Torch implementations of optimization algorithms."""

    adam = torch.optim.Adam
    sgd = torch.optim.SGD
    lbfgs = torch.optim.LBFGS

    # File contains how each backend should setup the optimizers and do the
    # optimization step.
    # This is then used in the training controller to do the optimization step for each
    # backend in a unified way.

    @staticmethod
    def setup_optimizer(optimization_phase, trainer):
        torch_params = []
        for param in trainer.trainable_parameters:
            torch_params.append(param.parameters)

        optimizer_obj = optimization_phase.optimizer(
            torch_params,  # type: ignore
            optimization_phase.lr.value,
            **optimization_phase.optimizer_args,
        )
        return optimizer_obj

    @staticmethod
    def do_optimization_step(
        optimization_phase,
        eval_function: Callable,
        step_idx: int,
        train_state,
    ):
        if optimization_phase.optimizer.requires_closure:
            optimization_phase.optimizer_obj.step(
                lambda: TorchOptimBackend._closure(
                    optimization_phase, eval_function, step_idx, train_state
                )
            )
            total_loss = TorchOptimBackend._closure(
                optimization_phase, eval_function, step_idx, train_state
            )
        else:
            total_loss = TorchOptimBackend._closure(
                optimization_phase, eval_function, step_idx, train_state
            )
            optimization_phase.optimizer_obj.step()

        train_state.total_train_loss = total_loss.item()  # type: ignore

    @staticmethod
    def _closure(
        optimization_phase,
        eval_function: Callable,
        step_idx: int,
        train_state,
    ):
        from qewton.optim.base import EvaluationPhase

        optimization_phase.optimizer_obj.zero_grad()
        eval_function(step_idx)

        total_loss = 0.0
        for loss_value in train_state.losses[EvaluationPhase.TRAIN].values():
            total_loss += loss_value

        total_loss.backward()  # type: ignore
        return total_loss

    @staticmethod
    def _cleanup():
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
