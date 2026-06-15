from typing import Any
import numpy as np
from qewton.backends.math import MathBackend
from qewton.backends.torch.device import get_torch_device
from qewton.config.devices import Device, cpu


class NumpyMathBackend(MathBackend[np.ndarray]):

    add = np.add
    multiply = np.multiply
    all = np.all
