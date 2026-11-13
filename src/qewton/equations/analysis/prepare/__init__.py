"""Prepare: check a problem and bring it into the form the other passes expect.

`validation.py` checks what only shows across constraints, `normalization.py`
rewrites the problem (time lifting, residual form), `infer.py` infers undeclared
output shapes, and `signatures.py` says what each function is.
"""

from qewton.equations.analysis.prepare.infer import Inference, infer_shapes
from qewton.equations.analysis.prepare.normalization import Normalized, normalize
from qewton.equations.analysis.prepare.signatures import Signature, Signatures
from qewton.equations.analysis.prepare.validation import validate

__all__ = [
    "Inference",
    "infer_shapes",
    "Normalized",
    "normalize",
    "Signature",
    "Signatures",
    "validate",
]
