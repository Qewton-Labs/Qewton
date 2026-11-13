"""Analysis of a task: checks, structure, classification and method suggestions.

`AnalysisContext(task)` (`context.py`) runs the passes on demand and keeps each
result. The passes are plain functions, grouped by what they find out:

- `prepare/`: validate, normalize, infer shapes, signatures;
- `structure/`: derivative orders, operator/source split, coupling graph, batching;
- `classify/`: equation and block types, condition kinds and counts, function spaces;
- `suggest/`: solution methods, one `Rule` subclass each;
- `training.py`: how each unknown can be learned; `overview.py`: what a problem holds.
"""
