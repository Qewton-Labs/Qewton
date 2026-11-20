from __future__ import annotations

import shutil
import subprocess

from dataclasses import dataclass
from functools import cache

from qewton.config.saving.loading import Deserializer
from qewton.config.saving.saving import Serializable, Serializer
from qewton.config.saving.schema_keys import SavingKeys


def cuda_available():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class Device(Serializable):
    """A device data and computations live on. Saved by its kind (and
    index) and loaded as the same canonical object, `cpu` or `cuda(i)`."""

    def save(self, serializer: Serializer) -> None:
        serializer.set_serialization_data(
            id(self),
            {
                SavingKeys.KEY_TYPE: SavingKeys.KEY_SERIALIZABLE,
                SavingKeys.KEY_CLASS: self.__class__.__name__,
                SavingKeys.KEY_MODULE: self.__class__.__module__,
                "index": getattr(self, "index", None),
            },
        )

    @classmethod
    def construct_new_object(cls, serializer: Deserializer, data_config: dict) -> Device:
        return cpu if cls is CPU else cuda(data_config["index"])

    def ready_to_load(self, serializer: Deserializer, data_config: dict) -> bool:
        return True

    def load(self, serializer: Deserializer, data_config: dict) -> None:
        pass  # the canonical object is complete


@dataclass(frozen=True)
class CPU(Device):
    pass


@dataclass(frozen=True)
class CUDA(Device):
    index: int


@cache
def cuda(index: int) -> CUDA:
    return CUDA(index)


cpu = CPU()
