import shutil
import subprocess

from dataclasses import dataclass
from functools import cache


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
class Device:
    pass


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
