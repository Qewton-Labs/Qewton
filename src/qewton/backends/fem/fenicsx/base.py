from qewton.backends.fem.base import FEMBackend


from .form import FenicsXFormBackend


class FenicsXBackend(FEMBackend):
    """A Backend that implements all operations related to FEniCSx."""

    form = FenicsXFormBackend
