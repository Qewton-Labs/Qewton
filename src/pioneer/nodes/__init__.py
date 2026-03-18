import importlib.util

from .base import Node, Port
from .operations.slice_nodes import SliceNode, SplitNode
from .operations.normalization import NormalizationNode, InverseNormalizationNode

# Pytorch classes (only import when Pytorch is available)
if importlib.util.find_spec("torch") is not None:
    from .operations.gradient_nodes import GradientTrackingNode
