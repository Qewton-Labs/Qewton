from ...trainer.training_controllers import LogEntry


class TuningState:

    def __init__(self, save_path: str) -> None:
        self.finished_trials: int = 0

        self.tune_history: list[list[LogEntry]] = []
        self.stop_tuning: bool = False

        self.save_path = save_path

    def add_trial_history(self, trial_history: list[LogEntry]):
        # TODO: maybe reduce resolution of history
        self.tune_history.append(trial_history)
