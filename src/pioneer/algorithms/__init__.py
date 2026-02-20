import importlib.util

from .base import AlgorithmNode, AlgorithmState

# Pytorch classes (only import when Pytorch is available)
if importlib.util.find_spec("torch") is not None:
    from .pytorch_algorithms.fcn import TorchFCN
    from .pytorch_algorithms.pcann import TorchPCANN
    from .pytorch_algorithms.wrapper import PyTorchWrapper

# Tensorflow classes
if importlib.util.find_spec("tensorflow") is not None:
    from .tensorflow_fcn import TFFCN

# TorchPhysics classes
if importlib.util.find_spec("torchphysics") is not None:
    from .pytorch_algorithms.tp_models import TorchPhysicsFNO
