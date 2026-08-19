from .base import Backend, TensorType, DeepLearningBackend, ComputingBackend
from .numpy.base import NumPyBackend

# internal hierarchy: if torch exists, we set it as the default backend,
# otherwise if tensorflow exists, we set it as the default backend.
_backend_found = False

try:
    from .torch.base import TorchBackend

    DEFAULT_DL_BACKEND = TorchBackend
    _backend_found = True
except (ImportError, AttributeError):
    # Torch is not installed or failed to initialize (e.g. circular imports)
    pass

if not _backend_found:
    try:
        from .tensorflow.base import TensorflowBackend

        DEFAULT_DL_BACKEND = TensorflowBackend
        _backend_found = True
    except (ImportError, AttributeError):
        pass

if not _backend_found:
    DEFAULT_DL_BACKEND = DeepLearningBackend


def _concrete_backends(cls=ComputingBackend):
    """Recursively walks ComputingBackend's subclasses, yielding only the
    concrete ones (those that actually declare their own default_dtype -
    Backend.default_dtype is an unset ClassVar, so abstract intermediates
    like DeepLearningBackend never match)."""
    for sub in cls.__subclasses__():
        if "default_dtype" in vars(sub):
            yield sub
        yield from _concrete_backends(sub)


def resolve_backend(tensor) -> type[ComputingBackend]:
    """Finds the ComputingBackend subclass whose default_dtype matches the
    given tensor's type.

    Raises if no backend matches, or if more than one does. The latter can
    in principle happen if two backends wrap the same underlying tensor type
    (e.g. a hypothetical SciPy backend also using plain np.ndarray, same as
    NumPyBackend) - the tensor's type alone can't disambiguate that, so this
    fails loudly with both candidates named rather than silently guessing.
    Callers that already know their backend (e.g. via a Node/Geometry) should
    use it directly instead of inferring it from a tensor.
    """
    matches = [b for b in _concrete_backends() if isinstance(tensor, b.default_dtype)]
    if not matches:
        raise ValueError(
            f"No known backend recognizes tensors of type {type(tensor).__name__!r}."
        )
    if len(matches) > 1:
        names = ", ".join(b.__name__ for b in matches)
        raise ValueError(
            f"Ambiguous backend for tensor of type {type(tensor).__name__!r}: "
            f"{names} all declare it as their default_dtype. Pass the backend "
            "explicitly instead of inferring it from the tensor."
        )
    return matches[0]
