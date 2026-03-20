import importlib.util

from .base import Trainer

# Pytorch classes (only import when Pytorch is available)
if importlib.util.find_spec("torch") is not None:
    from .pytorch_trainer import PyTorchTrainer


# Tensorflow classes
if importlib.util.find_spec("tensorflow") is not None:
    from .tensorflow_trainer import TensorFlowTrainer
