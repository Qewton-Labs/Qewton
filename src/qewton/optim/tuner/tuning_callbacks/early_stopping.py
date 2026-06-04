from .tuning_callback import TuningCallback
from ...base import EvaluationPhase
from ...trainer.training_controllers import TrainerState
from ....constraints.base import Constraint


class EarlyStoppingTuneCallback(TuningCallback):
    """
    A tuning callback that implements early stopping based on a specified
    constraint and threshold.
    If the monitored metric is worse then the average of the best percentile
    of the previous trials, this trial will be stopped early.
    """

    def __init__(
        self,
        monitor_constraint: Constraint,
        top_k: int = 10,
        persistence: int = 3,
        history_to_start: int = 10,
        check_interval: int = 100,
        priority: int = -5,
    ):
        """
        Initializes the EarlyStoppingCallback.

        Args:
            monitor_constraint (Constraint): The constraint to monitor for early
                stopping.
            top_k (int): The number of previous trials to consider for early stopping.
            persistence (int): The number of consecutive checks that must indicate
                stopping before actually stopping.
            history_to_start (int): The minimum number of completed trials before
                early stopping can be applied.
            check_interval (int): The number of iterations between early stopping checks.
            priority (int): The priority of this callback (lower means it will be
                called earlier).
        """
        super().__init__(priority=priority)
        self.monitor = monitor_constraint
        self.top_k = top_k
        self.history_to_start = max(
            history_to_start, top_k
        )  # ensure we have enough history to start
        self.check_interval = check_interval
        self.persistence = persistence
        self.best_value = None

        self.check_counter = 0
        self.last_early_stop_idx = -1
        self.early_stopping_history = []

    def on_training_start(self, state: TrainerState):
        # Check if we have enough trials to start with early stopping
        if len(self.tune_state.tune_history) < self.history_to_start:  # type: ignore
            return

        # Go through all previous trials
        for trial in self.tune_state.tune_history:  # type: ignore
            last_log_idx = 0
            for log in trial:
                current_idx = int(log.iteration / self.check_interval)
                if current_idx > last_log_idx:
                    last_log_idx = current_idx
                    if len(self.early_stopping_history) < last_log_idx:
                        self.early_stopping_history.append([])

                    if self.monitor.name in log.losses[EvaluationPhase.VALIDATION]:
                        self.early_stopping_history[last_log_idx - 1].append(
                            log.losses[EvaluationPhase.VALIDATION][self.monitor.name]
                        )
                    elif self.monitor.name in log.losses[EvaluationPhase.TRAIN]:
                        self.early_stopping_history[last_log_idx - 1].append(
                            log.losses[EvaluationPhase.TRAIN][self.monitor.name]
                        )

        # Now compute the average of the best k values for each check point
        for idx, values in enumerate(self.early_stopping_history):
            if len(values) > self.top_k:
                values.sort()
                best_k_values = values[: self.top_k]
                avg_best_k = sum(best_k_values) / self.top_k
                self.early_stopping_history[idx] = avg_best_k
            else:
                # Else add a large value so we continue
                self.early_stopping_history[idx] = 1000000

    def on_train_step_end(self, state: TrainerState):
        if (
            state.iteration % self.check_interval == 0
            and self.last_early_stop_idx + 1 < len(self.early_stopping_history)
        ):
            self.last_early_stop_idx += 1
            if self.monitor.name in state.losses[EvaluationPhase.VALIDATION]:
                current_value = state.losses[EvaluationPhase.VALIDATION][
                    self.monitor.name
                ]
            elif self.monitor.name in state.losses[EvaluationPhase.TRAIN]:
                current_value = state.losses[EvaluationPhase.TRAIN][self.monitor.name]
            else:
                return
            print(current_value, self.early_stopping_history[self.last_early_stop_idx])
            if current_value > self.early_stopping_history[self.last_early_stop_idx]:
                self.check_counter += 1
                print("here")
                if self.check_counter >= self.persistence:
                    state.stop_training_timer("Early stopping triggered.")
            else:
                self.check_counter = 0
