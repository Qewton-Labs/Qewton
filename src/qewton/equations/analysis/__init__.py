"""Analysis of a task: checks, structure, classification and method suggestions.

`AnalysisContext(task)` (`context.py`) runs the passes on demand and keeps each
result. The passes are plain functions, grouped by what they find out:

- `prepare/`: validation, normalization, shape inference, signatures;
- `structure/`: derivative orders, operator/source split, coupling graph, batching;
- `classification/`: equation and block types, condition kinds, function spaces;
- `suggestions/`: solution methods, one `Rule` subclass each;
- `training.py`: how each unknown can be learned;
- `problem_overview.py`: what a problem holds, before any task.
"""

# Each subpackage lists its public names in its own __all__.
# pylint: disable=wildcard-import,unused-wildcard-import
from qewton.equations.analysis import classification, prepare, structure, suggestions
from qewton.equations.analysis.classification import *
from qewton.equations.analysis.context import (
    AnalysisContext,
    batch_placement,
    batching,
    classify,
    coupling_graph,
    decoupled_axes,
    inherited_batch_axes,
    signatures,
    split,
    training_modes,
)
from qewton.equations.analysis.prepare import *
from qewton.equations.analysis.problem_overview import Overview, overview
from qewton.equations.analysis.structure import *
from qewton.equations.analysis.suggestions import *
from qewton.equations.analysis.training import TrainingMode

__all__ = [
    "AnalysisContext",
    "batch_placement",
    "batching",
    "classify",
    "coupling_graph",
    "decoupled_axes",
    "inherited_batch_axes",
    "signatures",
    "split",
    "training_modes",
    "Overview",
    "overview",
    "TrainingMode",
]
__all__ += prepare.__all__
__all__ += structure.__all__
__all__ += classification.__all__
__all__ += suggestions.__all__
