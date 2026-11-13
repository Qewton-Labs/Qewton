from qewton.backends.fem.base import FEMBackend


from qewton.backends.fem.fenicsx.form import FenicsXFormBackend
from qewton.backends.fem.fenicsx.function_space import FenicsXFunctionSpaceBackend
from qewton.backends.numpy.base import NumPyBackend


class FenicsXBackend(FEMBackend):
    """A Backend that implements all operations related to FEniCSx."""

    form = FenicsXFormBackend
    comp = NumPyBackend
    function_space = FenicsXFunctionSpaceBackend
