import importlib.util

from .base import Node, Port, _NodeRuntime
from .control_nodes import ControlNode
from .operations.slice_nodes import SliceNode, SplitNode
from .operations.normalization import NormalizationNode, InverseNormalizationNode

from .datasets.base import DataSet


# Pytorch classes (only import when Pytorch is available)
if importlib.util.find_spec("torch") is not None:
    from .operations.gradient_nodes import GradientTrackingNode
    from .datasets.pytorch_dataset import TorchDataSet
