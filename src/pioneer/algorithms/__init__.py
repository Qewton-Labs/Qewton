import importlib.util

from .base import AlgorithmNode, AlgorithmState

# Pytorch classes (only import when Pytorch is available)
if importlib.util.find_spec("torch") is not None:
    from .pytorch_algorithms.pytorch_test import TorchFCN
    from .pytorch_algorithms.pcann_test import TorchPCANN

# Tensorflow classes
if importlib.util.find_spec("tensorflow") is not None:
    from .tensorflow_test import TFFCN
