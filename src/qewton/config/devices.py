from dataclasses import dataclass
from functools import cache


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
