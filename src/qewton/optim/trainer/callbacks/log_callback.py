import os
import csv
from pathlib import Path

from .base_callback import Callback
from ..training_controllers import TrainerState
from ...base import EvaluationPhase


class LogCallback(Callback):
    def __init__(
        self,
        file_name: str = "training_logs",
        log_phase: EvaluationPhase = EvaluationPhase.ALWAYS,
        log_interval=-1,
        priority=-100,
    ):
        """
        If log_interval is -1 will use the same interval as used for the history saving
        in the trainer.
        Args:
            log_interval (int) :
        """
        super().__init__(priority)
        self.log_interval = log_interval
        self.file_name = file_name
        self.log_phase = log_phase

    @property
    def saves_data(self) -> bool:
        return True

    def on_training_start(self, state: TrainerState):
        if not state.enable_logging:
            raise RuntimeError(
                "Saving of train history, requires enable_logging=True in the trainer."
            )
        if self.log_interval <= 0:
            self.log_interval = state.log_interval
        if self.log_interval < state.log_interval:
            print(
                "Note: A smaller log interval in the callback will save duplicate data \
                points. Increase the log interval of this callback or decrease the interval \
                of the trainer."
            )

    def on_train_step_end(self, state: TrainerState):
        if state.iteration % self.log_interval == 0:
            pass

    def flatten_log_dict(self, state: TrainerState) -> dict[str, float | None]:
        if len(state.history) == 0:
            return {}

        last_entry = state.history[-1]

        logs: dict[str, float | None] = {
            "iteration": last_entry.iteration,
        }
        for phase, losses in last_entry.losses.items():
            if self.log_phase not in (phase, EvaluationPhase.ALWAYS):
                continue
            for name, value in losses.items():
                logs[f"{phase.name.lower()}/{name}"] = value

        return logs


class CSVLogger(LogCallback):

    def __init__(
        self,
        file_name: str = "training_logs",
        log_phase: EvaluationPhase = EvaluationPhase.ALWAYS,
        log_interval=-1,
        priority=-100,
    ):
        super().__init__(file_name, log_phase, log_interval, priority)
        self.path: Path
        self.header_written = False

    def on_training_start(self, state: TrainerState):
        super().on_training_start(state)

        filepath = os.path.join(state.save_path, self.file_name)
        self.path = Path(filepath)
        if self.path.suffix != ".csv":
            self.path = self.path.with_suffix(".csv")

        os.makedirs(state.save_path, exist_ok=True)

    def on_train_step_end(self, state: TrainerState):
        if state.iteration % self.log_interval == 0:
            logs = self.flatten_log_dict(state)
            if len(logs) == 0:  # no logs found yet
                return

            write_header = not self.path.exists() or not self.header_written

            with open(self.path, "a", newline="", encoding="utf-8") as f:

                writer = csv.DictWriter(f, fieldnames=logs.keys())

                if write_header:
                    writer.writeheader()
                    self.header_written = True

                writer.writerow(logs)


class TensorboardLogger(LogCallback):

    def __init__(
        self,
        file_name: str = "training_logs",
        log_phase: EvaluationPhase = EvaluationPhase.ALWAYS,
        log_interval=-1,
        priority=-100,
    ):
        super().__init__(file_name, log_phase, log_interval, priority)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.sum_writer_class = SummaryWriter
        except ImportError as e:
            raise ImportError("Pytorch required for using the TensorboardLogger.") from e
        self.tensorboard_writer: SummaryWriter

    def on_training_start(self, state: TrainerState):
        super().on_training_start(state)
        filepath = os.path.join(state.save_path, self.file_name)
        self.tensorboard_writer = self.sum_writer_class(filepath)

    def on_train_step_end(self, state: TrainerState):
        if state.iteration % self.log_interval == 0:

            logs = self.flatten_log_dict(state)
            if len(logs) == 0:  # no logs found yet
                return

            step = state.history[-1].iteration

            for key, value in logs.items():

                if key == "iteration":
                    continue

                self.tensorboard_writer.add_scalar(key, value, step)

    def on_training_end(self, state: TrainerState):
        super().on_training_end(state)
        self.tensorboard_writer.close()
