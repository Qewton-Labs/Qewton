import importlib.util

from .base import AlgorithmNode, AlgorithmState

# Pytorch classes (only import when Pytorch is available)
if importlib.util.find_spec("torch") is not None:
    from .pytorch_test import TorchFCN

# Tensorflow classes
if importlib.util.find_spec("tensorflow") is not None:
    from .tensorflow_test import TFFCN
